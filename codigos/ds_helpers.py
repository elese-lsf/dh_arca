"""Ayudas para las clases en vivo del programa Data Scientist · caso NovaRetail.

Se importa desde los notebooks con:  from ds_helpers import *
Todo usa datos sintéticos: ningún resultado representa una situación real de ARCA.
"""
import html
import random
import warnings
import threading
import time
from pathlib import Path

import ipywidgets as W
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")   # salidas limpias al proyectar

AQUI = Path(__file__).parent
EVIDENCIAS = AQUI / "evidencias"

pd.options.display.float_format = lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.1f}"
pd.options.display.max_columns = 30

VIOLETA, VERDE, NARANJA, ROJO = "#7C5CF6", "#10D9A0", "#F59E0B", "#F87171"
RESULTADOS = {"intentos": 0, "aciertos": 0}

TARGET = "abandono_90d"
FEATURES_NUM = ["edad", "antiguedad_meses", "compras_12m", "ticket_promedio", "dias_desde_ultima_compra",
                "contactos_soporte_90d", "usa_app", "descuento_promedio", "satisfaccion"]
FEATURES_CAT = ["canal", "segmento", "region"]
FEATURES = FEATURES_NUM + FEATURES_CAT
SEMILLA = 42


# ----------------------------------------------------------------------
# presentación
# ----------------------------------------------------------------------
def _caja(titulo, cuerpo="", color=VIOLETA):
    cuerpo_html = f"<div style='margin-top:4px'>{cuerpo}</div>" if cuerpo else ""
    return (f"<div style='border-left:5px solid {color};padding:8px 12px;margin:6px 0;"
            f"border-radius:4px;background:rgba(124,92,246,0.08)'>"
            f"<div style='font-weight:600'>{titulo}</div>{cuerpo_html}</div>")


def _reporte(titulo, checks):
    filas = "".join(f"<div style='padding:2px 0'>{'✅' if ok else '❌'} {texto}</div>" for ok, texto in checks)
    todo = all(ok for ok, _ in checks)
    display(HTML(_caja(f"{'🎉' if todo else '🔎'} {titulo}", filas, VERDE if todo else NARANJA)))
    return todo


def aviso(titulo, cuerpo="", color=VIOLETA):
    display(HTML(_caja(titulo, cuerpo, color)))


# ----------------------------------------------------------------------
# datos
# ----------------------------------------------------------------------
_DATOS = {"df": None, "universo": None}
COLUMNAS_ESPERADAS = ["id_cliente", "fecha_alta", "canal", "segmento", "region", "edad", "antiguedad_meses", "compras_12m",
                      "ticket_promedio", "dias_desde_ultima_compra", "contactos_soporte_90d", "usa_app",
                      "descuento_promedio", "satisfaccion", "motivo_baja", "abandono_90d"]


def _generador_de_practica():
    import sys
    sys.path.insert(0, str(AQUI / "src"))
    from novaretail.generador import generar_novaretail
    return generar_novaretail()


def usar_datos(df_propio=None):
    """Define qué datos usa la clase.

    - Con un DataFrame propio (el que genera TU código): lo valida y lo usa en todas las ayudas y verificadores.
    - Con None: usa el generador de práctica del paquete.
    """
    _DATOS["universo"] = None
    if df_propio is None:
        _DATOS["df"] = None
        aviso("Datos de práctica", "Se usa el generador de práctica del paquete (datos sintéticos de NovaRetail).", VIOLETA)
        return True
    faltan = [c for c in COLUMNAS_ESPERADAS if c not in df_propio.columns]
    if faltan:
        aviso("Faltan columnas en tus datos",
              "Los notebooks y los verificadores esperan estas columnas: <code>" + ", ".join(faltan) + "</code>. "
              "Corré de nuevo la celda de generación de datos, que ya trae el código completo.", NARANJA)
        return False
    _DATOS["df"] = df_propio.copy()
    aviso("Datos de la clase", f"{len(df_propio):,} filas · {df_propio['id_cliente'].nunique():,} clientes únicos · generados por la celda anterior.", VERDE)
    return True


def tablas(mostrar=True):
    """Todas las tablas del caso NovaRetail, generadas juntas y coherentes entre sí.

    Atributos: clientes, transacciones_crudas, transacciones, tiendas, unidades, precio, ejemplo_sucio, transacciones_500.
    Si usaste usar_datos(df_propio), las transacciones se construyen a partir de tus clientes.
    """
    if _DATOS["universo"] is None:
        import sys
        sys.path.insert(0, str(AQUI / "src"))
        from novaretail.generador import generar_universo
        _DATOS["universo"] = generar_universo(clientes=_DATOS["df"])
    u = _DATOS["universo"]
    if mostrar:
        display(u.describir())
    return u


def cargar(mostrar=True):
    """Devuelve los datos de NovaRetail (los propios si hiciste usar_datos(df); si no, el generador de práctica)."""
    df = _DATOS["df"].copy() if _DATOS["df"] is not None else _generador_de_practica()
    if mostrar:
        aviso("Datos cargados",
              f"{len(df):,} filas · {df.shape[1]} columnas · {df['id_cliente'].nunique():,} clientes únicos · "
              f"abandono_90d en la tabla cruda: {df[TARGET].mean():.1%}")
    return df


def limpiar(df):
    """Limpieza de referencia, con una decisión documentada por columna (sin ajustar nada con el target)."""
    d = df.drop_duplicates(subset="id_cliente").copy()
    d["edad"] = d["edad"].where(d["edad"].between(18, 100))
    d["region"] = d["region"].fillna("Sin dato")
    d["satisfaccion_faltante"] = d["satisfaccion"].isna().astype(int)
    return d.reset_index(drop=True)


def _preprocesador(num=None, cat=None):
    num = num or FEATURES_NUM
    cat = cat or FEATURES_CAT
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Sin dato")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def preprocesador(num=None, cat=None):
    """Preprocesamiento reutilizable: imputa y escala dentro del Pipeline, así que se ajusta solo con entrenamiento."""
    return _preprocesador(num, cat)


def particion(df=None, test_size=0.25):
    """Partición estratificada estándar de las clases 2 a 5: misma semilla, mismas filas de prueba."""
    d = limpiar(df if df is not None else cargar(mostrar=False))
    X, y = d[FEATURES], d[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEMILLA)


def tasa_base(y):
    return float(np.mean(y))


def metricas(y_true, proba, umbral):
    pred = (np.asarray(proba) >= umbral).astype(int)
    return {"umbral": umbral,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "contactados": int(pred.sum())}


# ----------------------------------------------------------------------
# interactividad (misma lógica que las otras clases)
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
            display(HTML(_caja("✅ Acertaste" if ok else f"❌ La respuesta era: {html.escape(str(correcta))}",
                               html.escape(explicacion), VERDE if ok else NARANJA)))

    btn.on_click(revelar)
    display(W.VBox([titulo, rb, btn, out]))


def marcador():
    aviso(f"🎯 Predicciones: {RESULTADOS['aciertos']} aciertos de {RESULTADOS['intentos']}")


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
    estado = {"resta": total, "corre": False}
    visor = W.HTML()
    play = W.Button(description="Iniciar", button_style="success", icon="play")
    pausa = W.Button(description="Pausar", icon="pause")
    reinicio = W.Button(description="Reiniciar", icon="refresh")

    def pintar():
        m, s = divmod(estado["resta"], 60)
        color = ROJO if estado["resta"] <= 30 else (NARANJA if estado["resta"] <= 120 else "inherit")
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
            threading.Thread(target=bucle, daemon=True).start()

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


ROLES = ["Conduce", "Implementa", "Valida", "Presenta"]


def _armar(nombres, tam, semilla=None):
    nombres = [n.strip() for n in nombres if n.strip()]
    random.Random(semilla).shuffle(nombres)
    n = len(nombres)
    if n == 0:
        return []
    k = max(1, -(-n // tam))
    equipos = [[] for _ in range(k)]
    for i, nom in enumerate(nombres):
        equipos[i % k].append(nom)
    return equipos


def _tabla_equipos(equipos, rotacion):
    tarjetas = []
    for i, eq in enumerate(equipos, 1):
        celdas = "".join(f"<li><b>{html.escape(p)}</b> · {ROLES[(j + rotacion) % len(ROLES)]}</li>" for j, p in enumerate(eq))
        tarjetas.append(f"<div style='border:1px solid {VIOLETA};border-radius:8px;padding:8px 14px;margin:6px;"
                        f"display:inline-block;vertical-align:top;min-width:190px'>"
                        f"<div style='font-weight:700;color:{VIOLETA}'>Equipo {i}</div><ul style='margin:4px 0'>{celdas}</ul></div>")
    return "".join(tarjetas)


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


def pausa_registro():
    """Cada equipo deja una decisión confirmada, una duda y un riesgo; las dudas se agrupan."""
    guardado = {"Decisión confirmada": [], "Duda": [], "Riesgo": []}
    equipo = W.Text(placeholder="Equipo", layout=W.Layout(width="110px"))
    tipo = W.Dropdown(options=list(guardado), layout=W.Layout(width="190px"))
    texto = W.Text(placeholder="Escribilo en una frase", layout=W.Layout(width="380px"))
    btn = W.Button(description="Agregar", button_style="primary", icon="plus")
    out = W.Output()

    def pintar():
        out.clear_output()
        with out:
            cols = []
            for t, items in guardado.items():
                lis = "".join(f"<li>{html.escape(x)}</li>" for x in items) or "<li style='opacity:.5'>—</li>"
                cols.append(f"<div style='display:inline-block;vertical-align:top;min-width:250px;margin-right:16px'>"
                            f"<div style='font-weight:700;color:{VIOLETA}'>{t} ({len(items)})</div><ul>{lis}</ul></div>")
            display(HTML("".join(cols)))

    def agregar(_):
        if texto.value.strip():
            et = f"{equipo.value.strip()}: " if equipo.value.strip() else ""
            guardado[tipo.value].append(et + texto.value.strip())
            texto.value = ""
            pintar()

    btn.on_click(agregar)
    pintar()
    display(W.VBox([W.HBox([equipo, tipo, texto, btn]), out]))


def guardar_evidencia(equipo, decision, poblacion, procedimiento, resultado, validacion, limite, extra=None):
    """Ficha de evidencia: decisión, población, procedimiento, resultado, validación y límite."""
    EVIDENCIAS.mkdir(exist_ok=True)
    campos = {"Decisión": decision, "Población y período": poblacion, "Procedimiento": procedimiento,
              "Resultado": resultado, "Validación": validacion, "Límite": limite}
    ruta = EVIDENCIAS / f"evidencia_{equipo.strip().lower().replace(' ', '_')}.md"
    texto = [f"# Ficha de evidencia · {equipo}", ""]
    for k, v in campos.items():
        texto += [f"## {k}", str(v), ""]
    for k, v in (extra or {}).items():
        texto += [f"## {k}", str(v), ""]
    ruta.write_text("\n".join(texto), encoding="utf-8")
    ok = _reporte("Ficha de evidencia", [(bool(str(v).strip()), k) for k, v in campos.items()])
    display(HTML(f"<div style='opacity:.7'>Guardada en <code>{ruta.relative_to(AQUI)}</code></div>"))
    return ok


def revision_cruzada_form(operacion=False, condicion=False):
    rev = W.Text(description="Revisa", placeholder="Equipo revisor")
    pres = W.Text(description="Presentó", placeholder="Equipo que presentó")
    ancho = W.Layout(width="560px")
    tecnica = W.Textarea(description="Técnica", placeholder="Una pregunta técnica", layout=ancho)
    negocio = W.Textarea(description="Negocio", placeholder="Una pregunta de negocio", layout=ancho)
    riesgo = W.Textarea(description="Riesgo", placeholder="Una pregunta sobre riesgo", layout=ancho)
    oper = W.Textarea(description="Operación", placeholder="Una pregunta de operación", layout=ancho)
    fort = W.Textarea(description="Fortaleza", placeholder="Una fortaleza", layout=ancho)
    brecha = W.Textarea(description="Brecha", placeholder="Una brecha prioritaria", layout=ancho)
    mejora = W.Textarea(description="Mejora", placeholder="Una mejora verificable", layout=ancho)
    cond = W.Textarea(description="Condición", placeholder="Una condición de aprobación", layout=ancho)
    btn = W.Button(description="Guardar devolución", button_style="primary", icon="save")
    out = W.Output()

    def guardar(_):
        out.clear_output()
        with out:
            EVIDENCIAS.mkdir(exist_ok=True)
            ruta = EVIDENCIAS / f"revision_{rev.value.strip().lower().replace(' ', '_') or 'sin_nombre'}.md"
            ruta.write_text(
                f"# Revisión cruzada\n\nRevisa: {rev.value}\nPresentó: {pres.value}\n\n"
                f"## Pregunta técnica\n{tecnica.value}\n\n## Pregunta de negocio\n{negocio.value}\n\n## Pregunta sobre riesgo\n{riesgo.value}\n\n"
                f"## Pregunta de operación\n{oper.value}\n\n"
                f"## Fortaleza\n{fort.value}\n\n## Brecha prioritaria\n{brecha.value}\n\n## Mejora verificable\n{mejora.value}\n\n"
                f"## Condición de aprobación\n{cond.value}\n",
                encoding="utf-8")
            _reporte("Devolución escrita", [
                (all(x.value.strip() for x in ((tecnica, negocio, riesgo) + ((oper,) if operacion else ()))),
                 "Pregunta técnica, de negocio y sobre riesgo" + (" y de operación" if operacion else "")),
                (all(x.value.strip() for x in (fort, brecha) + ((cond,) if condicion else (mejora,))),
                 "Fortaleza, brecha prioritaria y " + ("condición de aprobación" if condicion else "mejora verificable")),
                (True, f"Guardada en {ruta.relative_to(AQUI)}")])

    btn.on_click(guardar)
    campos = [tecnica, negocio, riesgo] + ([oper] if operacion else []) + [fort, brecha] + ([cond] if condicion else [mejora])
    display(W.VBox([W.HBox([rev, pres])] + campos + [btn, out]))


def cierre_individual():
    ancho = W.Layout(width="560px", height="60px")
    a = W.Textarea(description="Repetiría", layout=ancho)
    b = W.Textarea(description="Verificaría", layout=ancho)
    c = W.Textarea(description="No llevaría", placeholder="…todavía a producción", layout=ancho)
    btn = W.Button(description="Listo", button_style="success", icon="check")
    out = W.Output()

    def listo(_):
        out.clear_output()
        with out:
            aviso("Mi ticket de salida",
                  f"<b>Repetiría:</b> {html.escape(a.value)}<br><b>Verificaría:</b> {html.escape(b.value)}"
                  f"<br><b>No llevaría todavía a producción:</b> {html.escape(c.value)}", VERDE)

    btn.on_click(listo)
    display(W.VBox([a, b, c, btn, out]))


def comparar_tablas(a, b, etiquetas=("A", "B"), filas=10):
    def bloque(et, d):
        return (f"<div style='display:inline-block;vertical-align:top;margin-right:20px'>"
                f"<div style='font-weight:700;color:{VIOLETA}'>{et} · {len(d):,} filas</div>"
                f"{d.head(filas).to_html(index=False, border=0)}</div>")
    display(HTML(bloque(etiquetas[0], a) + bloque(etiquetas[1], b)))


# ----------------------------------------------------------------------
# CLASE 1 · verificadores
# ----------------------------------------------------------------------
def _ref_universo():
    return limpiar(cargar(mostrar=False))


def verificar_auditoria(df):
    """Tabla de auditoría: variable, defecto, decisión y validación."""
    if not isinstance(df, pd.DataFrame):
        aviso("Todavía no hay tabla", "Armá un DataFrame con las columnas variable, defecto, decision y validacion.", NARANJA)
        return False
    d = df.copy()
    d.columns = [str(c).lower().replace("ó", "o") for c in d.columns]
    req = ["variable", "defecto", "decision", "validacion"]
    faltan = [c for c in req if c not in d.columns]
    checks = [(not faltan, "Columnas: variable, defecto, decision, validacion" + (f" (faltan: {', '.join(faltan)})" if faltan else ""))]
    if faltan:
        return _reporte("Tabla de auditoría", checks)
    checks.append((len(d) >= 5, f"Al menos cinco filas auditadas (tiene {len(d)})"))
    checks.append((bool(d[req].astype(str).apply(lambda s: s.str.strip().str.len() > 2).all().all()),
                   "Ninguna celda está vacía: cada defecto tiene decisión y validación"))
    texto = " ".join(d["variable"].astype(str).str.lower()) + " " + " ".join(d["defecto"].astype(str).str.lower())
    cobertura = {
        "duplicados (id_cliente repetido)": "id_cliente" in texto or "duplic" in texto,
        "faltantes informativos (satisfaccion)": "satisfaccion" in texto or "satisfacción" in texto,
        "valores imposibles (edad)": "edad" in texto,
        "fuga (motivo_baja)": "motivo_baja" in texto or "fuga" in texto,
    }
    for k, ok in cobertura.items():
        checks.append((ok, f"Cubre {k}"))
    return _reporte("Tabla de auditoría", checks)


def verificar_universo(df):
    """El universo no puede alterarse por accidente al limpiar."""
    if not isinstance(df, pd.DataFrame):
        aviso("Todavía no hay datos limpios", "Guardá tu DataFrame limpio en una variable y pasalo al verificador.", NARANJA)
        return False
    ref = _ref_universo()
    n_ref, tasa_ref = len(ref), ref[TARGET].mean()
    checks = [
        ("id_cliente" in df.columns and df["id_cliente"].is_unique, "Un cliente por fila: id_cliente sin repetidos"),
        (0.99 * n_ref <= len(df) <= n_ref,
         f"El universo se conserva: {n_ref:,} clientes únicos (tu tabla tiene {len(df):,})"
         + (f" — perdiste {n_ref - len(df)} filas ({(n_ref - len(df)) / n_ref:.1%}): es tolerable solo si es una decisión que medís y documentás"
            if len(df) < n_ref and len(df) >= 0.99 * n_ref else "")),
    ]
    if TARGET in df.columns and len(df):
        tasa = df[TARGET].mean()
        ok = abs(tasa - tasa_ref) <= 0.005
        msg = f"La tasa de abandono no cambió por la limpieza: {tasa_ref:.1%} contra {tasa:.1%}"
        if not ok:
            msg += " — ¿borraste filas que no eran al azar? Medí qué tipo de cliente se fue."
        checks.append((ok, msg))
    else:
        checks.append((False, f"La tabla incluye la columna {TARGET}"))
    return _reporte("Universo y tasa de abandono", checks)


def verificar_variable(serie, nombre="variable"):
    """Chequea que una variable nueva sea alineable, completa y sin fuga evidente."""
    ref = _ref_universo()
    if not isinstance(serie, pd.Series):
        aviso("Pasá una Series de pandas", "Por ejemplo: df['compras_por_mes'].", NARANJA)
        return False
    s = serie.reset_index(drop=True)
    y = ref[TARGET].reset_index(drop=True)
    checks = [(len(s) == len(ref), f"Alineada con el universo: {len(ref):,} filas (la variable tiene {len(s):,})")]
    if len(s) != len(ref):
        return _reporte(f"Variable «{nombre}»", checks)
    faltantes = float(s.isna().mean())
    checks.append((faltantes < 0.30, f"Faltantes de la variable: {faltantes:.1%} (se espera menos del 30%)"))
    valida = s.notna()
    try:
        x = pd.to_numeric(s[valida], errors="coerce") if s.dtype != object else pd.factorize(s[valida])[0]
        auc = roc_auc_score(y[valida], x)
        auc = max(auc, 1 - auc)
    except Exception:
        auc = float("nan")
    checks.append((not (auc > 0.90), f"Señal contra el abandono: AUC aislado de {auc:.2f}"
                   + (" — una variable sola no debería separar casi perfecto: revisá si usa el target o algo posterior." if auc > 0.90 else "")))
    corr = float(np.corrcoef(pd.to_numeric(s[valida], errors="coerce").fillna(0), y[valida])[0, 1]) if s.dtype != object else 0.0
    checks.append((abs(corr) < 0.95, f"No es una copia del target (correlación {corr:.2f})"))
    return _reporte(f"Variable «{nombre}»", checks)


def verificar_mini_analisis(tabla):
    """Abandono por canal y segmento: tasa y tamaño de cada grupo."""
    if not isinstance(tabla, pd.DataFrame):
        aviso("Todavía no hay tabla", "Armá un DataFrame con canal, segmento, clientes y tasa_abandono.", NARANJA)
        return False
    t = tabla.copy()
    t.columns = [str(c).lower() for c in t.columns]
    req = ["canal", "segmento", "clientes", "tasa_abandono"]
    faltan = [c for c in req if c not in t.columns]
    checks = [(not faltan, "Columnas: canal, segmento, clientes, tasa_abandono" + (f" (faltan: {', '.join(faltan)})" if faltan else ""))]
    if faltan:
        return _reporte("Mini análisis de abandono", checks)
    ref = _ref_universo().groupby(["canal", "segmento"]).agg(clientes=(TARGET, "size"), tasa_abandono=(TARGET, "mean")).reset_index()
    checks.append((len(t) == len(ref), f"Un grupo por canal y segmento: {len(ref)} (tu tabla tiene {len(t)})"))
    m = ref.merge(t[req], on=["canal", "segmento"], how="left", suffixes=("_ref", ""))
    ok_n = bool(m["clientes"].notna().all() and np.allclose(m["clientes"], m["clientes_ref"]))
    checks.append((ok_n, "El tamaño de cada grupo coincide con el universo limpio (un cliente, una fila)"))
    tasas = pd.to_numeric(t["tasa_abandono"], errors="coerce")
    if tasas.between(0, 1).all():
        ok_t = bool(m["tasa_abandono"].notna().all() and np.allclose(m["tasa_abandono"], m["tasa_abandono_ref"], atol=0.005))
        checks.append((ok_t, "Las tasas de abandono coinciden con el universo limpio"))
    else:
        checks.append((False, "La tasa es una fracción entre 0 y 1 (no un porcentaje de 0 a 100)"))
    return _reporte("Mini análisis de abandono", checks)


# ----------------------------------------------------------------------
# CLASE 2 · verificadores y referencias
# ----------------------------------------------------------------------
def datos_umbrales():
    """Predicciones y probabilidades de un modelo base, para la práctica de umbrales."""
    from sklearn.linear_model import LogisticRegression
    X_tr, X_te, y_tr, y_te = particion()
    pipe = Pipeline([("pre", _preprocesador()), ("clf", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
    return y_te.to_numpy(), pipe.predict_proba(X_te)[:, 1]


def verificar_umbrales(resultado, u1=0.5, u2=0.25):
    """resultado: dict {umbral: {'precision':..,'recall':..,'f1':..}} o DataFrame con umbral/precision/recall/f1."""
    y, p = datos_umbrales()
    if isinstance(resultado, pd.DataFrame):
        r = {float(row["umbral"]): row for _, row in resultado.iterrows()}
    elif isinstance(resultado, dict):
        r = {float(k): v for k, v in resultado.items()}
    else:
        aviso("Todavía no hay resultado", "Pasá un diccionario por umbral o un DataFrame con umbral, precision, recall y f1.", NARANJA)
        return False
    checks = []
    for u in (u1, u2):
        if u not in r:
            checks.append((False, f"Falta el umbral {u}"))
            continue
        ref = metricas(y, p, u)
        try:
            ok = all(abs(float(r[u][k]) - ref[k]) <= 0.005 for k in ("precision", "recall", "f1"))
            msg = f"Umbral {u}: precisión, recall y F1 coinciden con las predicciones dadas"
        except (TypeError, ValueError, KeyError):
            ok, msg = False, f"Umbral {u}: completá precisión, recall y F1 con números (quedan TODO)"
        checks.append((ok, msg))
    return _reporte("Práctica de umbrales", checks)


def verificar_fuga(columnas):
    cols = [str(c).lower() for c in columnas]
    fuga = [c for c in cols if c in ("motivo_baja", TARGET)]
    return _reporte("Validación de fuga", [
        (not fuga, "Ninguna columna del modelo contiene información posterior al abandono"
         + (f" (sacá: {', '.join(fuga)})" if fuga else ""))])


def verificar_comparacion(tabla, n_test_esperado=None):
    """Tabla comparativa de tres modelos con el mismo split."""
    if not isinstance(tabla, pd.DataFrame):
        aviso("Todavía no hay tabla", "Armá un DataFrame con modelo, auc, precision, recall, f1 y n_test.", NARANJA)
        return False
    t = tabla.copy()
    t.columns = [str(c).lower() for c in t.columns]
    req = ["modelo", "auc", "precision", "recall", "f1", "n_test"]
    faltan = [c for c in req if c not in t.columns]
    checks = [(not faltan, "Columnas: modelo, auc, precision, recall, f1, n_test" + (f" (faltan: {', '.join(faltan)})" if faltan else ""))]
    if faltan:
        return _reporte("Comparación de modelos", checks)
    n_ref = n_test_esperado or len(particion()[3])
    checks.append((len(t) >= 3, f"Al menos tres modelos comparados (hay {len(t)})"))
    checks.append((bool((t["n_test"] == n_ref).all()), f"Todos se evaluaron con el mismo split: {n_ref:,} clientes de prueba"))
    checks.append((bool(t["auc"].between(0.5, 0.99).all()), "Los AUC son plausibles (entre 0,5 y 0,99): un AUC casi perfecto suele indicar fuga"))
    nombres = " ".join(t["modelo"].astype(str).str.lower())
    checks.append((("log" in nombres) and ("arbol" in nombres or "árbol" in nombres or "tree" in nombres)
                   and any(k in nombres for k in ("forest", "boost", "ensamble", "xgb")),
                   "Incluye regresión logística, árbol y un ensamble"))
    return _reporte("Comparación de modelos", checks)


# ----------------------------------------------------------------------
# CLASE 3 · verificadores
# ----------------------------------------------------------------------
PROBLEMAS_ARQUITECTURA = {
    "Clasificar fotos de productos dañados": "CNN",
    "Predecir la próxima compra a partir de la secuencia de las últimas 12 semanas": "RNN",
    "Clasificar y resumir reclamos escritos por clientes": "Transformer",
    "Predecir el abandono con la tabla de clientes de NovaRetail": "Modelo clásico",
}


def mapa_arquitecturas():
    """Cuatro problemas, cuatro desplegables: el equipo elige y revela."""
    ops = ["CNN", "RNN", "Transformer", "Modelo clásico"]
    dds = {p: W.Dropdown(options=[""] + ops, value="", description="", layout=W.Layout(width="170px")) for p in PROBLEMAS_ARQUITECTURA}
    out = W.Output()
    btn = W.Button(description="Revelar", button_style="primary", icon="eye")

    def revelar(_):
        out.clear_output()
        with out:
            checks = [(dds[p].value == c, f"{p}: {c}") for p, c in PROBLEMAS_ARQUITECTURA.items() if dds[p].value]
            if len(checks) < len(dds):
                display(HTML("<i>Elegí una arquitectura para los cuatro problemas.</i>"))
                return
            _reporte("Mapa de arquitecturas", checks)

    btn.on_click(revelar)
    filas = [W.HBox([W.HTML(f"<div style='width:520px'>{html.escape(p)}</div>"), dds[p]]) for p in PROBLEMAS_ARQUITECTURA]
    display(W.VBox(filas + [btn, out]))


def guardar_ficha(equipo, titulo, campos, requeridos, prefijo="ficha"):
    """Guarda una ficha de trabajo en evidencias/ y controla que estén completos los campos requeridos."""
    EVIDENCIAS.mkdir(exist_ok=True)
    ruta = EVIDENCIAS / f"{prefijo}_{equipo.strip().lower().replace(' ', '_')}.md"
    texto = [f"# {titulo} · {equipo}", ""]
    for k in requeridos:
        texto += [f"## {k.replace('_', ' ').capitalize()}", str(campos.get(k, "")), ""]
    ruta.write_text("\n".join(texto), encoding="utf-8")
    ok = _reporte(titulo, [(bool(str(campos.get(k, "")).strip()), k.replace("_", " ").capitalize()) for k in requeridos])
    display(HTML(f"<div style='opacity:.7'>Guardado en <code>{ruta.relative_to(AQUI)}</code></div>"))
    return ok


def guardar_canvas(equipo, **campos):
    """Canvas de experimento (clase 3)."""
    req = ["aplicacion", "linea_base", "arquitectura", "split", "criterio_detencion", "metrica", "costo",
           "revision_etica", "condicion_abandono"]
    return guardar_ficha(equipo, "Canvas de experimento", campos, req, "canvas")


def guardar_ruta_despliegue(equipo, **campos):
    """Ruta notebook → paquete → API → registro → dashboard (clase 4)."""
    req = ["ruta", "versionado", "prueba_de_humo", "rollback", "latencia", "drift", "responsable_humano"]
    return guardar_ficha(equipo, "Ruta de despliegue", campos, req, "despliegue")


# ----------------------------------------------------------------------
# CLASE 4 · auditoría del contrato y pruebas
# ----------------------------------------------------------------------
def auditar_contrato(ruta):
    """Revisa un contrato OpenAPI: tipos, obligatorios, rangos, versión, códigos de error y ejemplo."""
    import yaml
    with open(ruta, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    esquemas = c.get("components", {}).get("schemas", {})
    entrada = next((v for k, v in esquemas.items() if "entrada" in k.lower() or "request" in k.lower()), {})
    props = entrada.get("properties", {})
    numericos_esperados = ["edad", "antiguedad_meses", "compras_12m", "ticket_promedio", "dias_desde_ultima_compra",
                           "contactos_soporte_90d", "usa_app", "descuento_promedio", "satisfaccion"]
    sin_tipo = [k for k, v in props.items() if "type" not in v]
    mal_tipo = [k for k in numericos_esperados if props.get(k, {}).get("type") == "string"]
    numericos = [k for k, v in props.items() if v.get("type") in ("integer", "number")]
    sin_rango = [k for k in numericos if "minimum" not in props[k] and "maximum" not in props[k] and "enum" not in props[k]]
    obligatorios = set(entrada.get("required", []))
    faltan_oblig = [k for k in props if k not in obligatorios]
    paths = c.get("paths", {})
    ruta_ver = any("/v1" in p or "/v2" in p for p in paths)
    version = bool(c.get("info", {}).get("version")) and ruta_ver
    codigos = set()
    for p in paths.values():
        for op in p.values():
            codigos |= {str(k) for k in op.get("responses", {})}
    tiene_ejemplo = "example" in entrada or any("example" in v for v in props.values()) or \
        any("example" in str(op.get("requestBody", {})) for p in paths.values() for op in p.values())
    cierra = entrada.get("additionalProperties") is False
    return _reporte("Auditoría del contrato", [
        (bool(props) and not sin_tipo and not mal_tipo, "Tipos: todos los campos declaran su tipo y los numéricos no son texto"
         + (f" (sin tipo: {', '.join(sin_tipo)})" if sin_tipo else "") + (f" (numérico declarado como texto: {', '.join(mal_tipo)})" if mal_tipo else "")),
        (bool(props) and not faltan_oblig, "Campos obligatorios: todos los que necesita el modelo están en `required`"
         + (f" (no obligatorios: {', '.join(faltan_oblig)})" if faltan_oblig else "")),
        (bool(numericos) and not sin_rango, "Rangos: los numéricos tienen mínimo y máximo" + (f" (sin rango: {', '.join(sin_rango)})" if sin_rango else "")),
        (cierra, "Campos extra: el esquema rechaza campos desconocidos (`additionalProperties: false`)"),
        (version, "Versión: el contrato declara versión y la ruta incluye /v1"),
        ({"200", "422"} <= codigos and bool(codigos & {"400", "500", "503"}), f"Códigos de error: definidos {sorted(codigos)}; se esperan 200, 422 y al menos un error del servidor"),
        (tiene_ejemplo, "Ejemplo: hay al menos un ejemplo de pedido"),
    ])


def correr_pruebas():
    """Ejecuta pytest sobre tests/ y muestra el resultado."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-p", "no:warnings", str(AQUI / "tests" / "test_api.py")],
                       capture_output=True, text=True, cwd=AQUI)
    ok = r.returncode == 0
    display(HTML(_caja("✅ Pruebas aprobadas" if ok else "❌ Hay pruebas que fallan",
                       f"<pre style='margin:0;white-space:pre-wrap'>{html.escape(r.stdout[-1800:])}</pre>", VERDE if ok else ROJO)))
    return ok


def psi(esperado, actual, bins=10):
    """Population Stability Index entre dos muestras de una variable numérica."""
    esperado, actual = np.asarray(esperado, float), np.asarray(actual, float)
    cortes = np.unique(np.quantile(esperado[~np.isnan(esperado)], np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado[~np.isnan(esperado)], cortes)[0] / max(1, (~np.isnan(esperado)).sum())
    a = np.histogram(actual[~np.isnan(actual)], cortes)[0] / max(1, (~np.isnan(actual)).sum())
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ----------------------------------------------------------------------
# CLASE 5 · rúbrica y revisión
# ----------------------------------------------------------------------
RUBRICA = [
    ("Pregunta y decisión", "Responde la pregunta y la decisión definidas"),
    ("Población y objetivo", "Declara población, período, granularidad y variable objetivo"),
    ("Reproducibilidad", "Conserva una ruta reproducible desde los datos hasta la conclusión"),
    ("Métricas y referencia", "Compara contra una referencia y usa métricas acordes al costo del error"),
    ("Controles", "Incluye controles de calidad, fuga, sesgo y generalización"),
    ("Evidencia e interpretación", "Separa evidencia, interpretación y recomendación"),
    ("Límites y siguiente paso", "Documenta al menos una limitación y un siguiente paso"),
]


def rubrica_interactiva(titulo="Proyecto a revisar"):
    """Puntúa cada criterio (0 a 2) y devuelve un diagnóstico con la mejora prioritaria."""
    ops = [("No aparece", 0), ("Parcial", 1), ("Logrado", 2)]
    sel = {k: W.ToggleButtons(options=ops, value=0, style={"button_width": "92px"}) for k, _ in RUBRICA}
    btn = W.Button(description="Calcular devolución", button_style="primary", icon="calculator")
    out = W.Output()

    def calcular(_):
        out.clear_output()
        with out:
            total = sum(v.value for v in sel.values())
            maxi = 2 * len(RUBRICA)
            peores = sorted(RUBRICA, key=lambda kv: sel[kv[0]].value)[:1]
            mejores = [k for k, _ in RUBRICA if sel[k].value == 2][:1]
            aviso(f"{titulo}: {total} de {maxi} puntos",
                  f"<b>Fortaleza:</b> {mejores[0] if mejores else 'ninguna todavía'}<br>"
                  f"<b>Brecha prioritaria:</b> {peores[0][0]} — {peores[0][1]}", VERDE if total >= 11 else NARANJA)

    btn.on_click(calcular)
    filas = [W.HBox([W.HTML(f"<div style='width:330px'><b>{k}</b><br><small style='opacity:.7'>{d}</small></div>"), sel[k]]) for k, d in RUBRICA]
    display(W.VBox(filas + [btn, out]))


def heatmap_abandono(tabla):
    """Mapa de calor de la tasa de abandono por canal y segmento, con el tamaño de cada grupo."""
    import matplotlib.pyplot as plt
    if not isinstance(tabla, pd.DataFrame) or not {"canal", "segmento", "tasa_abandono", "clientes"} <= set(tabla.columns):
        aviso("Todavía no hay tabla para graficar",
              "Completá la tabla (canal, segmento, clientes, tasa_abandono) y volvé a correr esta celda.", NARANJA)
        return
    tasa = tabla.pivot(index="segmento", columns="canal", values="tasa_abandono")
    n = tabla.pivot(index="segmento", columns="canal", values="clientes")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(tasa.values, cmap="Purples", vmin=0, vmax=max(0.35, float(np.nanmax(tasa.values))))
    ax.set_xticks(range(len(tasa.columns)), tasa.columns)
    ax.set_yticks(range(len(tasa.index)), tasa.index)
    for i in range(tasa.shape[0]):
        for j in range(tasa.shape[1]):
            ax.text(j, i, f"{tasa.values[i, j]:.0%}\nn={int(n.values[i, j])}", ha="center", va="center",
                    color="white" if tasa.values[i, j] > 0.2 else "black", fontsize=10)
    ax.set_title("Abandono a 90 días por canal y segmento")
    fig.colorbar(im, ax=ax, label="tasa de abandono")
    plt.tight_layout()
    plt.show()


def explorar_umbral(y_true, proba, capacidad=300, costo_fp=5.0, costo_fn=60.0):
    """Explorador interactivo: mové el umbral y los costos y mirá matriz, métricas, capacidad y costo total."""
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    u = W.FloatSlider(value=0.25, min=0.05, max=0.90, step=0.01, description="Umbral", layout=W.Layout(width="360px"))
    cfp = W.FloatText(value=costo_fp, description="Costo FP", layout=W.Layout(width="170px"))
    cfn = W.FloatText(value=costo_fn, description="Costo FN", layout=W.Layout(width="170px"))
    out = W.Output()

    def pintar(*_):
        out.clear_output(wait=True)
        with out:
            pred = (proba >= u.value).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            costo = fp * cfp.value + fn * cfn.value
            contactados = int(pred.sum())
            display(pd.DataFrame([[tn, fp], [fn, tp]], index=["Real: no abandona", "Real: abandona"],
                                 columns=["Predice: no abandona", "Predice: abandona"]))
            ok_cap = contactados <= capacidad
            aviso(f"Umbral {u.value:.2f}: {contactados} contactos ({'dentro de' if ok_cap else '⚠ supera'} la capacidad de {capacidad})",
                  f"Precisión {prec:.1%} · recall {rec:.1%} · falsos positivos {fp} · falsos negativos {fn}<br>"
                  f"Costo total = {fp} × {cfp.value:g} + {fn} × {cfn.value:g} = <b>{costo:,.0f}</b>",
                  VERDE if ok_cap else NARANJA)

    for w in (u, cfp, cfn):
        w.observe(pintar, names="value")
    pintar()
    display(W.VBox([W.HBox([u, cfp, cfn]), out]))


def dibujar_red(capas=(24, 8, 1), nombres=("Entradas\n(24 variables)", "Capa oculta\n(8 neuronas, tanh)", "Salida\n(probabilidad)")):
    """Dibuja la anatomía de una red densa: entradas, capas, salida."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    xs = np.linspace(0.1, 0.9, len(capas))
    pos = []
    for x, n in zip(xs, capas):
        k = min(n, 8)
        ys = np.linspace(0.18, 0.82, k) if k > 1 else np.array([0.5])
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="#B9A3FF", lw=0.6, alpha=0.6, zorder=1)
    for x, col, nom, n in zip(xs, pos, nombres, capas):
        for (px, py) in col:
            ax.scatter(px, py, s=260, color="#7C5CF6", zorder=2)
        if n > 8:
            ax.text(x, 0.09, "⋮", ha="center", fontsize=14)
        ax.text(x, 0.97, nom, ha="center", va="top", fontsize=10)
    ax.text(0.5, 0.01, "Ciclo de entrenamiento: adelante → pérdida → atrás → actualizar", ha="center", fontsize=10, color="#6B6B85")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def verificar_lista(items, n, nombre):
    """Controla que haya al menos n elementos no vacíos (por ejemplo, pruebas o métricas operativas)."""
    llenos = [str(x).strip() for x in items if str(x).strip() and str(x).strip() != "..."]
    return _reporte(f"{nombre.capitalize()}", [(len(llenos) >= n, f"Al menos {n} {nombre} escritas (tenés {len(llenos)})")])


def simular_registro(dias=14, por_dia=120, drift_desde=9, semilla=3):
    """Registro SIMULADO de predicciones: tráfico a partir del conjunto de prueba, con un corrimiento de datos desde cierto día."""
    import sys
    sys.path.insert(0, str(AQUI / "src"))
    from novaretail.predictor import cargar_artefacto
    modelo = cargar_artefacto()
    _, X_te, _, _ = particion()
    rng = np.random.default_rng(semilla)
    filas = []
    for d in range(1, dias + 1):
        lote = X_te.sample(por_dia, random_state=int(rng.integers(0, 10_000))).copy()
        if d >= drift_desde:                          # el comportamiento cambia: los clientes tardan más en volver
            lote["dias_desde_ultima_compra"] = (lote["dias_desde_ultima_compra"] * 1.6 + 15).round()
        p = modelo.predict_proba(lote)[:, 1]
        lat = rng.gamma(4.0, 3.5, por_dia) + (rng.random(por_dia) < 0.03) * rng.uniform(80, 200, por_dia)
        for pi, li, dias_u in zip(p, lat, lote["dias_desde_ultima_compra"]):
            filas.append({"dia": d, "version_modelo": "1.0.0", "probabilidad": float(pi), "priorizar": int(pi >= 0.25),
                          "latencia_ms": float(li), "dias_desde_ultima_compra": float(dias_u)})
    return pd.DataFrame(filas)


def tablero_monitoreo(registro, referencia=None):
    """Tablero mínimo: volumen, latencia (p50 y p95), tasa de priorización y drift (PSI) por día."""
    import matplotlib.pyplot as plt
    if referencia is None:
        referencia = particion()[0]["dias_desde_ultima_compra"].to_numpy()
    g = registro.groupby("dia")
    vol = g.size()
    p50 = g["latencia_ms"].median()
    p95 = g["latencia_ms"].quantile(0.95)
    tasa = g["priorizar"].mean()
    drift = g["dias_desde_ultima_compra"].apply(lambda s: psi(referencia, s.to_numpy()))
    fig, ejes = plt.subplots(2, 2, figsize=(10, 5.4))
    ejes[0, 0].bar(vol.index, vol.values, color="#7C5CF6")
    ejes[0, 0].set_title("Predicciones por día (ventana: 1 día)")
    ejes[0, 1].plot(p50.index, p50.values, label="p50", color="#7C5CF6")
    ejes[0, 1].plot(p95.index, p95.values, label="p95", color="#F59E0B")
    ejes[0, 1].set_title("Latencia (ms)")
    ejes[0, 1].legend()
    ejes[1, 0].plot(tasa.index, tasa.values * 100, color="#10D9A0")
    ejes[1, 0].set_title("% de clientes priorizados")
    ejes[1, 1].plot(drift.index, drift.values, color="#F87171")
    ejes[1, 1].axhline(0.10, color="#888", ls="--", lw=0.8)
    ejes[1, 1].axhline(0.25, color="#888", ls="--", lw=0.8)
    ejes[1, 1].set_title("Drift de dias_desde_ultima_compra (PSI)")
    for ax in ejes.ravel():
        ax.set_xlabel("día")
    plt.tight_layout()
    plt.show()
    return pd.DataFrame({"predicciones": vol, "latencia_p95_ms": p95, "priorizados": tasa, "psi": drift})


CHECK_DEFENSA = [
    "La decisión de negocio está al inicio, antes del algoritmo",
    "Población, período y variable objetivo declarados",
    "Se comparó contra una línea base",
    "Se reportó más de una métrica y también el resultado negativo",
    "Se controló fuga, sesgo y generalización",
    "La explicación no se presenta como causalidad",
    "Hay límites, un siguiente paso y un responsable humano",
]


def checklist_defensa(items=None):
    """Checklist de una defensa: marcá lo que el proyecto cumple y mirá las brechas."""
    items = items or CHECK_DEFENSA
    cajas = [W.Checkbox(value=False, description=t, indent=False, layout=W.Layout(width="640px")) for t in items]
    btn = W.Button(description="Ver brechas", button_style="primary", icon="search")
    out = W.Output()

    def ver(_):
        out.clear_output()
        with out:
            faltan = [c.description for c in cajas if not c.value]
            aviso(f"{len(items) - len(faltan)} de {len(items)} cumplidos",
                  "<br>".join(f"• {f}" for f in faltan) if faltan else "No quedan brechas.", NARANJA if faltan else VERDE)

    btn.on_click(ver)
    display(W.VBox(cajas + [btn, out]))


def crear_ficha_proyecto(equipo):
    """Copia ficha-proyecto-final.md a evidencias/ para completarla en VS Code."""
    EVIDENCIAS.mkdir(exist_ok=True)
    destino = EVIDENCIAS / f"ficha_proyecto_{equipo.strip().lower().replace(' ', '_')}.md"
    if not destino.exists():
        destino.write_text((AQUI / "ficha-proyecto-final.md").read_text(encoding="utf-8"), encoding="utf-8")
    aviso("Ficha creada", f"Abrila y completala: <code>{destino.relative_to(AQUI)}</code>", VERDE)
    return destino


def guardar_entrega_final(equipo, **campos):
    req = ["presentacion_ejecutiva", "repositorio_o_notebook", "ficha_de_modelo", "matriz_de_riesgos", "recomendacion", "plan_de_monitoreo"]
    return guardar_ficha(equipo, "Entrega final", campos, req, "entrega")


def ordenar_ciclo():
    """Actividad breve: armar el orden lógico de trabajo del equipo de NovaRetail (CRISP-DM)."""
    correcto = ["Negocio", "Datos", "Preparación", "Modelo"]
    ops = ["", "Negocio", "Datos", "Preparación", "Modelo"]
    dds = [W.Dropdown(options=ops, value="", description=f"Paso {i + 1}", layout=W.Layout(width="230px")) for i in range(4)]
    btn = W.Button(description="Revisar", button_style="primary", icon="check")
    out = W.Output()

    def revisar(_):
        out.clear_output()
        with out:
            elegido = [d.value for d in dds]
            if "" in elegido or len(set(elegido)) < 4:
                display(HTML("<i>Elegí una tarjeta distinta para cada paso.</i>"))
                return
            ok = elegido == correcto
            aviso("✅ Ese es el orden lógico" if ok else "🔎 Probá de nuevo",
                  "Negocio → Datos → Preparación → Modelo. <b>Pero no es una fila de pasos:</b> es un ciclo. Si en Preparación aparece que la calidad "
                  "de los datos no alcanza, se vuelve a Datos o incluso a Negocio, y se redefine qué se puede decidir." if ok else
                  "Pensá qué hay que saber antes de tocar un dato, y qué hay que tener listo antes de modelar.", VERDE if ok else NARANJA)

    btn.on_click(revisar)
    display(W.VBox(dds + [btn, out]))


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("html", "random", "threading", "time", "Path")]
