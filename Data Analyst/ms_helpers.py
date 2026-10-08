"""Ayudas para la clase en vivo 1 · Mercado Sur (SQL y análisis de datos).

Se importa desde los notebooks con:  from ms_helpers import *
Todo corre en memoria con DuckDB: genera los datos desde código y entiende el mismo SQL de la clase.
"""
import html
import random
import threading
import time
from pathlib import Path

import duckdb
import ipywidgets as W
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import HTML, display

# números legibles al proyectar: sin notación científica
pd.options.display.float_format = lambda x: f"{x:,.2f}" if (abs(x) >= 1 or x == 0) else f"{x:.4f}"

AQUI = Path(__file__).parent
DATA_DIR = AQUI / "data"
EVIDENCIAS_DIR = AQUI / "evidencias"

# ----------------------------------------------------------------------
# CONFIGURACIÓN: si tus CSV reales tienen otros nombres, editá estos mapeos.
# Formato: {"nombre_en_tu_csv": "nombre_esperado"}
# ----------------------------------------------------------------------
MAPEO_COLUMNAS = {
    "clientes": {},
    "productos": {},
    "ventas": {},
}
# Si los estados vienen con otro texto, por ejemplo {"Completada": "completada"}
MAPEO_ESTADOS = {}

ESQUEMA = {
    "clientes": ["id_cliente", "region"],
    "productos": ["id_producto", "categoria"],
    "ventas": ["id_venta", "fecha", "id_cliente", "id_producto", "canal", "unidades",
               "precio_unitario", "descuento_pct", "estado", "satisfaccion"],
}
ESTADOS_ESPERADOS = ["completada", "devuelta", "cancelada"]

con = duckdb.connect(":memory:")
RESULTADOS = {"intentos": 0, "aciertos": 0}

VIOLETA, VERDE, NARANJA, ROJO = "#7C5CF6", "#10D9A0", "#F59E0B", "#F87171"

NETA = "unidades * precio_unitario * (1 - descuento_pct)"


# ----------------------------------------------------------------------
# utilidades de presentación
# ----------------------------------------------------------------------
def _caja(titulo, cuerpo="", color=VIOLETA):
    cuerpo_html = f"<div style='margin-top:4px'>{cuerpo}</div>" if cuerpo else ""
    return (f"<div style='border-left:5px solid {color};padding:8px 12px;margin:6px 0;"
            f"border-radius:4px;background:rgba(124,92,246,0.08)'>"
            f"<div style='font-weight:600'>{titulo}</div>{cuerpo_html}</div>")


def _reporte(titulo, checks):
    """checks: lista de (ok, texto). Devuelve True si todo está bien."""
    filas = "".join(
        f"<div style='padding:2px 0'>{'✅' if ok else '❌'} {texto}</div>" for ok, texto in checks)
    todo = all(ok for ok, _ in checks)
    color = VERDE if todo else NARANJA
    encabezado = f"{'🎉' if todo else '🔎'} {titulo}"
    display(HTML(_caja(encabezado, filas, color)))
    return todo


# ----------------------------------------------------------------------
# carga de datos y magia %%sql
# ----------------------------------------------------------------------
def _sql_magic(line, cell):
    var = (line or "").strip().split()[0] if (line or "").strip() else "df"
    t0 = time.perf_counter()
    try:
        res = con.sql(cell)
        df = res.df() if res is not None else None
    except Exception as e:  # el error de SQL se muestra en una caja, no rompe el notebook
        display(HTML(_caja("❌ Error de SQL", f"<code>{html.escape(str(e))}</code>", ROJO)))
        return None
    if df is None:
        return None
    ms = (time.perf_counter() - t0) * 1000
    get_ipython().user_ns[var] = df
    n = len(df)
    display(HTML(f"<div style='opacity:.65;font-size:12px'>{n:,} filas · {len(df.columns)} columnas · "
                 f"{ms:.0f} ms · resultado guardado en <code>{var}</code></div>"))
    display(df if n <= 40 else df.head(20))
    if n > 40:
        display(HTML("<i style='opacity:.65'>Se muestran las primeras 20 filas.</i>"))
    return None


def activar_magia():
    ip = get_ipython()
    if ip is not None:
        ip.register_magic_function(_sql_magic, "cell", "sql")


def _columnas(tabla):
    return list(con.sql(f"DESCRIBE {tabla}").df()["column_name"])


def usar_tablas(clientes, productos, ventas, mostrar=True):
    """Registra tres tablas (DataFrames) en DuckDB, activa %%sql y valida el esquema."""
    for nombre, df in (("clientes", clientes), ("productos", productos), ("ventas", ventas)):
        con.register("_tmp", df)
        reemplazos = [f"CAST({c} AS DATE) AS {c}" for c in ("fecha", "fecha_alta") if c in df.columns]
        extra = f" REPLACE ({', '.join(reemplazos)})" if reemplazos else ""
        con.execute(f"CREATE OR REPLACE TABLE {nombre} AS SELECT *{extra} FROM _tmp")
        con.unregister("_tmp")
    activar_magia()
    ok = _validar_esquema()
    if mostrar:
        resumen_tablas()
    return ok


def cargar_datos(data_dir=None, mostrar=True):
    """Prepara las tres tablas. Sin argumentos, las genera desde código (no hay archivos que subir).
    Con data_dir, lee los CSV de esa carpeta (por si querés usar datos reales) y aplica los mapeos."""
    d = Path(data_dir) if data_dir else DATA_DIR
    hay_csv = all((d / f"{t}.csv").exists() for t in ESQUEMA)
    if not hay_csv:
        if data_dir:
            raise FileNotFoundError(f"No encuentro los CSV en {d}. Copiá clientes.csv, productos.csv y ventas.csv.")
        import sys
        sys.path.insert(0, str(AQUI))
        from mercado_sur_datos import generar_mercado_sur
        t = generar_mercado_sur()
        return usar_tablas(t["clientes"], t["productos"], t["ventas"], mostrar=mostrar)
    for t in ESQUEMA:
        ruta = d / f"{t}.csv"
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv_auto('{ruta.as_posix()}', header=true)")
        for viejo, nuevo in MAPEO_COLUMNAS.get(t, {}).items():
            con.execute(f'ALTER TABLE {t} RENAME COLUMN "{viejo}" TO "{nuevo}"')
    for viejo, nuevo in MAPEO_ESTADOS.items():
        con.execute("UPDATE ventas SET estado = ? WHERE estado = ?", [nuevo, viejo])
    activar_magia()
    ok = _validar_esquema()
    if mostrar:
        resumen_tablas()
    return ok


def _validar_esquema():
    checks = []
    for t, cols in ESQUEMA.items():
        presentes = _columnas(t)
        faltan = [c for c in cols if c not in presentes]
        if faltan:
            checks.append((False, f"<b>{t}</b>: faltan las columnas <code>{', '.join(faltan)}</code>. "
                                  f"Editá <code>MAPEO_COLUMNAS</code> en ms_helpers.py."))
        else:
            n = con.sql(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            checks.append((True, f"<b>{t}</b>: {n:,} filas, columnas esperadas presentes"))
    if all(ok for ok, _ in checks):
        estados = sorted(con.sql("SELECT DISTINCT estado FROM ventas").df()["estado"].dropna().tolist())
        extra = [e for e in estados if e not in ESTADOS_ESPERADOS]
        checks.append((not extra, f"estados en <code>ventas</code>: <code>{', '.join(estados)}</code>"
                       + (" (esperados: completada, devuelta, cancelada; usá MAPEO_ESTADOS)" if extra else "")))
        mx = con.sql("SELECT MAX(descuento_pct) FROM ventas").fetchone()[0]
        checks.append((mx is not None and mx <= 1.0,
                       f"<code>descuento_pct</code> es una fracción de 0 a 1 (máximo: {mx})"
                       + ("" if mx is not None and mx <= 1.0 else ": si viene de 0 a 100, dividilo por 100")))
    return _reporte("Chequeo del entorno y de los datos", checks)


def resumen_tablas():
    partes = []
    for t in ESQUEMA:
        n = con.sql(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        desc = con.sql(f"DESCRIBE {t}").df()[["column_name", "column_type"]]
        cols = ", ".join(f"<code>{r.column_name}</code> <small style='opacity:.6'>{r.column_type}</small>"
                         for r in desc.itertuples())
        partes.append(f"<tr><td style='padding:4px 10px'><b>{t}</b></td>"
                      f"<td style='padding:4px 10px;text-align:right'>{n:,}</td>"
                      f"<td style='padding:4px 10px'>{cols}</td></tr>")
    display(HTML("<table style='border-collapse:collapse;font-size:13px'>"
                 "<tr><th style='text-align:left;padding:4px 10px'>Tabla</th>"
                 "<th style='padding:4px 10px'>Filas</th><th style='text-align:left;padding:4px 10px'>Columnas</th></tr>"
                 + "".join(partes) + "</table>"))


# ----------------------------------------------------------------------
# interactividad: predicciones, pistas, cronómetro, equipos, preguntas
# ----------------------------------------------------------------------
def predecir(pregunta, opciones, correcta, explicacion=""):
    titulo = W.HTML(f"<div style='font-size:16px;font-weight:600'>🔮 Predicción: {html.escape(pregunta)}</div>")
    rb = W.RadioButtons(options=opciones, value=None, layout=W.Layout(width="auto"))
    btn = W.Button(description="Revelar", button_style="primary", icon="eye")
    out = W.Output()

    def revelar(_):
        out.clear_output()
        with out:
            if rb.value is None:
                display(HTML("<i>Elegí una opción primero.</i>"))
                return
            ok = rb.value == correcta
            RESULTADOS["intentos"] += 1
            RESULTADOS["aciertos"] += int(ok)
            if ok:
                display(HTML(_caja("✅ Acertaste", html.escape(explicacion), VERDE)))
            else:
                display(HTML(_caja(f"❌ La respuesta era: {html.escape(str(correcta))}",
                                   html.escape(explicacion), NARANJA)))

    btn.on_click(revelar)
    display(W.VBox([titulo, rb, btn, out]))


def marcador():
    i, a = RESULTADOS["intentos"], RESULTADOS["aciertos"]
    display(HTML(_caja(f"🎯 Predicciones: {a} aciertos de {i}", "", VIOLETA)))


def pistas(lista):
    estado = {"n": 0}
    btn = W.Button(description=f"Pedir una pista (0/{len(lista)})", icon="lightbulb-o")
    out = W.Output()

    def click(_):
        if estado["n"] < len(lista):
            estado["n"] += 1
            with out:
                display(HTML(_caja(f"Pista {estado['n']}", lista[estado["n"] - 1], VIOLETA)))
            btn.description = f"Pedir una pista ({estado['n']}/{len(lista)})"
        if estado["n"] >= len(lista):
            btn.disabled = True

    btn.on_click(click)
    display(W.VBox([btn, out]))


def cronometro(minutos, titulo="Cronómetro"):
    total = int(minutos * 60)
    estado = {"resta": total, "corre": False, "hilo": None}
    visor = W.HTML()
    play = W.Button(description="Iniciar", button_style="success", icon="play")
    pausa = W.Button(description="Pausar", icon="pause")
    reinicio = W.Button(description="Reiniciar", icon="refresh")

    def pintar():
        m, s = divmod(estado["resta"], 60)
        color = ROJO if estado["resta"] <= 30 else ("#F59E0B" if estado["resta"] <= 120 else "inherit")
        visor.value = (f"<div style='font-size:13px;opacity:.7'>{html.escape(titulo)}</div>"
                       f"<div style='font-size:56px;font-weight:700;font-family:monospace;color:{color}'>{m:02d}:{s:02d}</div>")

    def bucle():
        while estado["corre"] and estado["resta"] > 0:
            time.sleep(1)
            if estado["corre"]:
                estado["resta"] -= 1
                pintar()
        estado["corre"] = False

    def iniciar(_):
        if not estado["corre"] and estado["resta"] > 0:
            estado["corre"] = True
            h = threading.Thread(target=bucle, daemon=True)
            estado["hilo"] = h
            h.start()

    def pausar(_):
        estado["corre"] = False

    def reiniciar(_):
        estado["corre"] = False
        estado["resta"] = total
        pintar()

    play.on_click(iniciar)
    pausa.on_click(pausar)
    reinicio.on_click(reiniciar)
    pintar()
    display(W.VBox([visor, W.HBox([play, pausa, reinicio])]))


ROLES = ["Conduce", "Ejecuta", "Valida", "Presenta"]


def _armar(nombres, tam, semilla=None):
    nombres = [n.strip() for n in nombres if n.strip()]
    rnd = random.Random(semilla)
    rnd.shuffle(nombres)
    n = len(nombres)
    if n == 0:
        return []
    k = max(1, -(-n // tam))  # techo: nunca más de `tam` personas por equipo
    equipos = [[] for _ in range(k)]
    for i, nom in enumerate(nombres):
        equipos[i % k].append(nom)
    return equipos


def _tabla_equipos(equipos, rotacion):
    filas = []
    for i, eq in enumerate(equipos, 1):
        celdas = "".join(
            f"<li><b>{html.escape(p)}</b> · {ROLES[(j + rotacion) % len(ROLES)]}</li>" for j, p in enumerate(eq))
        filas.append(f"<div style='border:1px solid {VIOLETA};border-radius:8px;padding:8px 14px;margin:6px;"
                     f"display:inline-block;vertical-align:top;min-width:190px'>"
                     f"<div style='font-weight:700;color:{VIOLETA}'>Equipo {i}</div><ul style='margin:4px 0'>{celdas}</ul></div>")
    return "".join(filas)


def armar_equipos(nombres, tam=4, semilla=None):
    """Arma equipos de 3 o 4 personas con roles rotativos."""
    equipos = _armar(list(nombres), tam, semilla)
    display(HTML(_tabla_equipos(equipos, 0)))
    return equipos


def equipos_interactivo():
    area = W.Textarea(placeholder="Un nombre por línea", layout=W.Layout(width="320px", height="140px"))
    tam = W.IntSlider(value=4, min=3, max=4, description="Tamaño", layout=W.Layout(width="260px"))
    armar = W.Button(description="Armar equipos", button_style="primary", icon="users")
    rotar = W.Button(description="Rotar roles", icon="refresh")
    out = W.Output()
    estado = {"equipos": [], "rot": 0}

    def pintar():
        out.clear_output()
        with out:
            display(HTML(_tabla_equipos(estado["equipos"], estado["rot"])))

    def on_armar(_):
        estado["equipos"] = _armar(area.value.split("\n"), tam.value)
        estado["rot"] = 0
        pintar()

    def on_rotar(_):
        estado["rot"] += 1
        pintar()

    armar.on_click(on_armar)
    rotar.on_click(on_rotar)
    display(W.VBox([W.HBox([area, W.VBox([tam, armar, rotar])]), out]))


def preguntas_pendientes():
    """Cada equipo deja una pregunta; se agrupan por concepto, herramienta o decisión."""
    categorias = ["Concepto", "Herramienta", "Decisión"]
    guardadas = {c: [] for c in categorias}
    equipo = W.Text(placeholder="Equipo", layout=W.Layout(width="120px"))
    texto = W.Text(placeholder="Pregunta pendiente", layout=W.Layout(width="360px"))
    cat = W.Dropdown(options=categorias, layout=W.Layout(width="140px"))
    btn = W.Button(description="Agregar", button_style="primary", icon="plus")
    out = W.Output()

    def pintar():
        out.clear_output()
        with out:
            cols = []
            for c in categorias:
                items = "".join(f"<li>{html.escape(t)}</li>" for t in guardadas[c]) or "<li style='opacity:.5'>—</li>"
                cols.append(f"<div style='display:inline-block;vertical-align:top;min-width:240px;margin-right:16px'>"
                            f"<div style='font-weight:700;color:{VIOLETA}'>{c} ({len(guardadas[c])})</div><ul>{items}</ul></div>")
            display(HTML("".join(cols)))

    def agregar(_):
        if texto.value.strip():
            etiqueta = f"{equipo.value.strip()}: " if equipo.value.strip() else ""
            guardadas[cat.value].append(etiqueta + texto.value.strip())
            texto.value = ""
            pintar()

    btn.on_click(agregar)
    pintar()
    display(W.VBox([W.HBox([equipo, texto, cat, btn]), out]))


# ----------------------------------------------------------------------
# validaciones de la demostración
# ----------------------------------------------------------------------
def validaciones_demo():
    n_ventas = con.sql("SELECT COUNT(*) FROM ventas").fetchone()[0]
    n_join = con.sql("SELECT COUNT(*) FROM ventas v JOIN productos p ON p.id_producto = v.id_producto").fetchone()[0]
    total_canales = con.sql(f"""SELECT SUM(f) FROM (SELECT SUM({NETA}) AS f FROM ventas
                                 WHERE estado = 'completada' GROUP BY canal)""").fetchone()[0]
    total_comp = con.sql(f"SELECT SUM({NETA}) FROM ventas WHERE estado = 'completada'").fetchone()[0]
    bruta = con.sql("SELECT SUM(unidades * precio_unitario) FROM ventas WHERE estado = 'completada'").fetchone()[0]
    _reporte("Tres validaciones de la consulta base", [
        (n_ventas == n_join, f"Filas antes del JOIN: {n_ventas:,} · después: {n_join:,} (el JOIN no duplica)"),
        (abs(total_canales - total_comp) < 0.01,
         f"La suma por canal ({total_canales:,.0f}) coincide con el total de completadas ({total_comp:,.0f})"),
        (total_comp <= bruta, f"La facturación neta ({total_comp:,.0f}) no supera a la bruta ({bruta:,.0f})"),
    ])


# ----------------------------------------------------------------------
# referencias y verificadores de ejercicios
# ----------------------------------------------------------------------
def _usuario(nombre, df):
    if df is not None:
        return df
    return get_ipython().user_ns.get(nombre)


def _ref_ej1():
    return con.sql(f"""
        SELECT v.canal, c.region,
               SUM(CASE WHEN month(v.fecha) <= 6 THEN {NETA} ELSE 0 END) AS facturacion_s1,
               SUM(CASE WHEN month(v.fecha) >  6 THEN {NETA} ELSE 0 END) AS facturacion_s2
        FROM ventas v JOIN clientes c ON c.id_cliente = v.id_cliente
        WHERE v.estado = 'completada'
        GROUP BY v.canal, c.region ORDER BY v.canal, c.region""").df()


def _total_neto(estados=None):
    filtro = "" if estados is None else f"WHERE estado IN ({', '.join(repr(e) for e in estados)})"
    return con.sql(f"SELECT SUM({NETA}) FROM ventas {filtro}").fetchone()[0]


def _diagnostico_total(total):
    comp, todas = _total_neto(["completada"]), _total_neto()
    if np.isclose(total, todas, rtol=1e-6):
        return "El total coincide con el de <b>todos los estados</b>: mezclaste completadas con devueltas y canceladas."
    if total > comp * 1.001:
        return "El total <b>supera</b> el de operaciones completadas: ¿el JOIN duplicó filas o falta el filtro de estado?"
    if total < comp * 0.999:
        return "El total es <b>menor</b> al de operaciones completadas: ¿algún filtro o un INNER JOIN con claves que no coinciden perdió filas?"
    return ""


def verificar_ej1(df=None):
    df = _usuario("ej1", df)
    if not isinstance(df, pd.DataFrame):
        display(HTML(_caja("Todavía no hay resultado", "Corré primero la celda de tu consulta (<code>%%sql ej1</code>).", NARANJA)))
        return False
    ref = _ref_ej1()
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    requeridas = ["canal", "region", "facturacion_s1", "facturacion_s2"]
    faltan = [c for c in requeridas if c not in df.columns]
    checks = [(not faltan, "Columnas pedidas: canal, region, facturacion_s1, facturacion_s2"
               + (f" (faltan: {', '.join(faltan)})" if faltan else ""))]
    if faltan:
        return _reporte("Ejercicio 1", checks)
    checks.append((len(df) == len(ref), f"Una fila por canal y región: esperadas {len(ref)}, tu consulta devolvió {len(df)}"))
    m = ref.merge(df[requeridas], on=["canal", "region"], how="left", suffixes=("_ref", ""))
    iguales = (len(m) == len(ref)
               and np.allclose(m["facturacion_s1"].fillna(-1), m["facturacion_s1_ref"], rtol=1e-6, atol=0.01)
               and np.allclose(m["facturacion_s2"].fillna(-1), m["facturacion_s2_ref"], rtol=1e-6, atol=0.01))
    checks.append((iguales, "Los valores de cada semestre coinciden con la facturación neta de operaciones completadas"))
    total = float(df["facturacion_s1"].sum() + df["facturacion_s2"].sum())
    if not iguales:
        diag = _diagnostico_total(total)
        if diag:
            checks.append((False, diag))
    ok = _reporte("Ejercicio 1: semestre contra semestre", checks)
    if ok:
        display(HTML(_caja("Para la puesta en común",
                           "Mostrá tu <b>definición</b> y <b>un control</b>. ¿Qué universo, qué grano, qué tratamiento de estados?", VIOLETA)))
    return ok


def _ref_segmentos():
    return con.sql(f"""
        SELECT v.canal, c.region, p.categoria,
               SUM(CASE WHEN v.estado = 'completada' AND month(v.fecha) <= 6 THEN {NETA} ELSE 0 END) AS fact_s1,
               SUM(CASE WHEN v.estado = 'completada' AND month(v.fecha) >  6 THEN {NETA} ELSE 0 END) AS fact_s2,
               SUM(CASE WHEN v.estado = 'completada' THEN {NETA} ELSE 0 END) AS facturacion_neta,
               COUNT(*) AS ops_total,
               COUNT(*) FILTER (WHERE v.estado = 'devuelta')   AS ops_devueltas,
               COUNT(*) FILTER (WHERE v.estado = 'completada') AS ops_completadas
        FROM ventas v
        JOIN clientes c  ON c.id_cliente  = v.id_cliente
        JOIN productos p ON p.id_producto = v.id_producto
        GROUP BY v.canal, c.region, p.categoria""").df()


COLS_CAMBIO = ("caida", "caída", "variacion", "variación", "dif", "delta", "cambio")


def verificar_desafio(df=None):
    df = _usuario("desafio", df)
    if not isinstance(df, pd.DataFrame):
        display(HTML(_caja("Todavía no hay resultado", "Corré primero la celda de tu consulta (<code>%%sql desafio</code>).", NARANJA)))
        return False
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    checks = [(len(df) <= 12, f"La tabla tiene como máximo 12 filas (tiene {len(df)})")]
    claves = ["canal", "region", "categoria"]
    faltan = [c for c in claves + ["tasa_devolucion"] if c not in df.columns]
    checks.append((not faltan, "Columnas: canal, region, categoria y tasa_devolucion"
                   + (f" (faltan: {', '.join(faltan)})" if faltan else "")))
    cambio = [c for c in df.columns if any(k in c for k in COLS_CAMBIO)]
    checks.append((bool(cambio), "Hay una columna que mide el cambio entre semestres (por ejemplo variacion_pct o caida)"))
    if faltan:
        return _reporte("Desafío por equipos", checks)
    checks.append((not df.duplicated(subset=claves).any(), "No hay segmentos repetidos (canal, región, categoría)"))
    tasa = pd.to_numeric(df["tasa_devolucion"], errors="coerce")
    checks.append((bool(tasa.between(0, 1).all()), "La tasa de devolución es una fracción entre 0 y 1 (no un porcentaje de 0 a 100)"))
    if tasa.between(0, 1).all() and len(df):
        ref = _ref_segmentos()
        ref["tasa_a"] = ref["ops_devueltas"] / ref["ops_total"]
        ref["tasa_b"] = ref["ops_devueltas"] / (ref["ops_devueltas"] + ref["ops_completadas"])
        m = df[claves + ["tasa_devolucion"]].merge(ref[claves + ["tasa_a", "tasa_b"]], on=claves, how="left")
        existe = bool(m["tasa_a"].notna().all())
        checks.append((existe, "Todos los segmentos existen en los datos"))
        if existe:
            t = pd.to_numeric(m["tasa_devolucion"])
            a = bool(np.allclose(t, m["tasa_a"], atol=0.005))
            b = bool(np.allclose(t, m["tasa_b"], atol=0.005))
            if a or b:
                universo = ("devueltas sobre todas las operaciones" if a
                            else "devueltas sobre completadas + devueltas")
                checks.append((True, f"La tasa coincide con una definición válida: <b>{universo}</b>. Declarala en tu evidencia."))
            else:
                checks.append((False, "La tasa no coincide con ninguna definición habitual "
                                      "(devueltas / todas, o devueltas / (completadas + devueltas)). ¿Qué universo usaste?"))
    return _reporte("Desafío por equipos", checks)


def _ref_ventana():
    return con.sql(f"""
        SELECT v.canal, c.region, p.categoria,
               SUM({NETA}) AS facturacion_neta,
               SUM({NETA}) / SUM(SUM({NETA})) OVER () AS participacion_total
        FROM ventas v
        JOIN clientes c  ON c.id_cliente  = v.id_cliente
        JOIN productos p ON p.id_producto = v.id_producto
        WHERE v.estado = 'completada'
        GROUP BY v.canal, c.region, p.categoria""").df()


def verificar_ventana(df=None):
    df = _usuario("bonus", df)
    if not isinstance(df, pd.DataFrame):
        display(HTML(_caja("Todavía no hay resultado", "Corré primero la celda del bonus (<code>%%sql bonus</code>).", NARANJA)))
        return False
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    ref = _ref_ventana()
    req = ["canal", "region", "categoria", "facturacion_neta", "participacion_total"]
    faltan = [c for c in req if c not in df.columns]
    checks = [(not faltan, "Columnas: canal, region, categoria, facturacion_neta y participacion_total"
               + (f" (faltan: {', '.join(faltan)})" if faltan else ""))]
    if faltan:
        return _reporte("Bonus: función de ventana", checks)
    checks.append((len(df) == len(ref), f"Conservaste el detalle: {len(ref)} segmentos (tu tabla tiene {len(df)})"))
    m = ref.merge(df[req], on=["canal", "region", "categoria"], how="left", suffixes=("_ref", ""))
    ok_val = bool(m["participacion_total"].notna().all() and
                  np.allclose(m["participacion_total"], m["participacion_total_ref"], atol=1e-9))
    checks.append((ok_val, "Cada segmento se compara contra el total (la participación suma 1)"))
    return _reporte("Bonus: función de ventana", checks)


# ----------------------------------------------------------------------
# comparación de soluciones (puesta en común)
# ----------------------------------------------------------------------
def comparar(sql_a, sql_b, etiquetas=("A", "B"), filas=12):
    try:
        a = con.sql(sql_a).df()
    except Exception as e:
        display(HTML(_caja(f"❌ Error en {etiquetas[0]}", f"<code>{html.escape(str(e))}</code>", ROJO)))
        return
    try:
        b = con.sql(sql_b).df()
    except Exception as e:
        display(HTML(_caja(f"❌ Error en {etiquetas[1]}", f"<code>{html.escape(str(e))}</code>", ROJO)))
        return

    def bloque(et, d):
        return (f"<div style='display:inline-block;vertical-align:top;margin-right:20px'>"
                f"<div style='font-weight:700;color:{VIOLETA}'>Solución {et} · {len(d):,} filas</div>"
                f"{d.head(filas).to_html(index=False, border=0, float_format=lambda x: f'{x:,.2f}')}</div>")

    display(HTML(bloque(etiquetas[0], a) + bloque(etiquetas[1], b)))
    mismo = a.shape == b.shape and list(a.columns) == list(b.columns)
    if mismo:
        try:
            iguales = np.allclose(a.select_dtypes("number").to_numpy(), b.select_dtypes("number").to_numpy(), rtol=1e-9)
        except Exception:
            iguales = False
    else:
        iguales = False
    _reporte("Comparación", [
        (a.shape[0] == b.shape[0], f"Mismas filas: {a.shape[0]:,} contra {b.shape[0]:,}"),
        (list(a.columns) == list(b.columns), "Mismas columnas"),
        (iguales, "Mismos valores numéricos"),
    ])


# ----------------------------------------------------------------------
# evidencia de salida y revisión cruzada
# ----------------------------------------------------------------------
def guardar_evidencia(equipo, pregunta, definicion, consulta, resultado=None, validaciones=(), limitacion=""):
    """Guarda la evidencia de salida: consulta final, definición, dos validaciones y una limitación."""
    EVIDENCIAS_DIR.mkdir(exist_ok=True)
    vals = [v for v in validaciones if str(v).strip()]
    ruta = EVIDENCIAS_DIR / f"evidencia_{equipo.strip().lower().replace(' ', '_')}.md"
    texto = [f"# Evidencia · {equipo}", "", "## Pregunta", pregunta, "", "## Definición de la métrica", definicion,
             "", "## Procedimiento (consulta final)", "```sql", consulta.strip(), "```", ""]
    if isinstance(resultado, pd.DataFrame):
        texto += ["## Resultado", "```", resultado.head(12).to_string(index=False), "```", ""]
    texto += ["## Validaciones"] + [f"{i}. {v}" for i, v in enumerate(vals, 1)]
    texto += ["", "## Límite (qué no se puede afirmar)", limitacion, ""]
    ruta.write_text("\n".join(texto), encoding="utf-8")
    ok = _reporte("Evidencia de salida", [
        (bool(str(pregunta).strip()), "Pregunta"),
        (bool(str(definicion).strip()), "Definición de la métrica"),
        (bool(str(consulta).strip()), "Consulta final"),
        (len(vals) >= 2, f"Dos validaciones (cargaste {len(vals)})"),
        (bool(str(limitacion).strip()), "Una limitación"),
    ])
    display(HTML(f"<div style='opacity:.7'>Guardada en <code>{ruta.relative_to(AQUI)}</code></div>"))
    return ok


def revision_cruzada_form():
    rev = W.Text(description="Revisa", placeholder="Equipo revisor")
    pres = W.Text(description="Presentó", placeholder="Equipo que presentó")
    metodo = W.Textarea(description="Método", placeholder="Una pregunta de método", layout=W.Layout(width="520px"))
    negocio = W.Textarea(description="Negocio", placeholder="Una pregunta de negocio", layout=W.Layout(width="520px"))
    solida = W.Textarea(description="Decisión sólida", placeholder="Una decisión sólida", layout=W.Layout(width="520px"))
    brecha = W.Textarea(description="Brecha", placeholder="Una brecha que se pueda corregir sin rehacer todo", layout=W.Layout(width="520px"))
    btn = W.Button(description="Guardar revisión", button_style="primary", icon="save")
    out = W.Output()

    def guardar(_):
        out.clear_output()
        with out:
            EVIDENCIAS_DIR.mkdir(exist_ok=True)
            ruta = EVIDENCIAS_DIR / f"revision_{rev.value.strip().lower().replace(' ', '_') or 'sin_nombre'}.md"
            ruta.write_text(
                f"# Revisión cruzada\n\nRevisa: {rev.value}\nPresentó: {pres.value}\n\n"
                f"## Pregunta de método\n{metodo.value}\n\n## Pregunta de negocio\n{negocio.value}\n\n"
                f"## Una decisión sólida\n{solida.value}\n\n## Una brecha corregible\n{brecha.value}\n", encoding="utf-8")
            completo = all(x.value.strip() for x in (metodo, negocio, solida, brecha))
            _reporte("Revisión cruzada", [(completo, "Pregunta de método, pregunta de negocio, decisión sólida y brecha"),
                                          (True, f"Guardada en {ruta.relative_to(AQUI)}")])

    btn.on_click(guardar)
    display(W.VBox([W.HBox([rev, pres]), metodo, negocio, solida, brecha, btn, out]))


def cierre_individual():
    a = W.Textarea(description="Voy a repetir", layout=W.Layout(width="520px", height="60px"))
    b = W.Textarea(description="Voy a verificar", layout=W.Layout(width="520px", height="60px"))
    c = W.Textarea(description="Voy a dejar de", layout=W.Layout(width="520px", height="60px"))
    btn = W.Button(description="Listo", button_style="success", icon="check")
    out = W.Output()

    def listo(_):
        out.clear_output()
        with out:
            display(HTML(_caja("Mi cierre", f"<b>Repetir:</b> {html.escape(a.value)}<br><b>Verificar:</b> {html.escape(b.value)}"
                                            f"<br><b>Dejar de hacer:</b> {html.escape(c.value)}", VERDE)))

    btn.on_click(listo)
    display(W.VBox([a, b, c, btn, out]))


def grafico_variacion(df=None):
    """Barras con la variación semestral por canal y región (a partir del resultado del ejercicio 1)."""
    import matplotlib.pyplot as plt
    df = _usuario("ej1", df)
    if not isinstance(df, pd.DataFrame) or not {"canal", "region", "facturacion_s1", "facturacion_s2"} <= set(df.columns):
        display(HTML(_caja("Todavía no hay datos para graficar",
                           "Completá y verificá el ejercicio 1 (<code>verificar_ej1()</code>) y volvé a correr esta celda.", NARANJA)))
        return
    d = df.copy()
    d["variacion"] = (d["facturacion_s2"] - d["facturacion_s1"]) / d["facturacion_s1"] * 100
    d["segmento"] = d["canal"] + " · " + d["region"]
    d = d.sort_values("variacion")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(d["segmento"], d["variacion"], color=[ROJO if v < 0 else VERDE for v in d["variacion"]])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Variación de la facturación neta, 2.º semestre contra 1.º (%)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()




def venn_joins():
    """Dibuja el clásico diagrama de Venn de los cuatro JOIN: qué filas se conservan cuando no hay pareja."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    casos = [("INNER JOIN", "solo las filas con pareja", "inner"), ("LEFT JOIN", "todas las de A", "left"),
             ("RIGHT JOIN", "todas las de B", "right"), ("FULL JOIN", "todas, de ambos lados", "full")]
    for ax, (nombre, desc, tipo) in zip(axes, casos):
        ax.set_xlim(0, 3.2); ax.set_ylim(0, 2.0); ax.set_aspect("equal"); ax.axis("off")
        if tipo == "inner":
            p = Circle((1.2, 1.0), 0.8, fc="#10D9A0", ec="none", alpha=0.85)
            ax.add_patch(p); p.set_clip_path(Circle((2.0, 1.0), 0.8, transform=ax.transData))
        if tipo in ("left", "full"):
            ax.add_patch(Circle((1.2, 1.0), 0.8, fc="#10D9A0", ec="none", alpha=0.85))
        if tipo in ("right", "full"):
            ax.add_patch(Circle((2.0, 1.0), 0.8, fc="#10D9A0", ec="none", alpha=0.85))
        ax.add_patch(Circle((1.2, 1.0), 0.8, fc="none", ec="#7C5CF6", lw=2))
        ax.add_patch(Circle((2.0, 1.0), 0.8, fc="none", ec="#7C5CF6", lw=2))
        ax.text(0.75, 1.0, "A", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(2.45, 1.0, "B", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_title(nombre, fontsize=11, fontweight="bold", color="#7C5CF6")
        ax.text(1.6, -0.05, desc, ha="center", va="top", fontsize=9)
    plt.tight_layout()
    plt.show()


__all__ = [n for n in dir() if not n.startswith("_") and n not in
           ("html", "random", "threading", "time", "Path", "duckdb", "get_ipython")] + ["con"]
