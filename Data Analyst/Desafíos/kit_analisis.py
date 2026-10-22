# ═══════════════════════════ KIT DE ANÁLISIS · versión completa ═══════════════════════════
# Una herramienta por paso del procedimiento, y cuatro extras para empezar con cualquier archivo:
#   cargar(ruta)            → lee un CSV o un Excel y convierte a fecha las columnas que lo parecen
#   auditoria(df)           → lista de hallazgos de calidad, cada uno con la decisión que hay que tomar
#   reporte_html(df, ruta)  → un reporte en un solo archivo HTML, con tablas y gráficos
#   exportar_excel(...)     → las tablas del reporte en un libro de Excel
#   reporte_completo(...)   → todo lo anterior en una llamada
# Los pasos del procedimiento (reporte_inicial, cuantificar_cambio, descartar_impostores, cortes, procedimiento)
# son los mismos del kit de la clase. El kit calcula; el criterio decide.
import base64, html, io, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

def describir(serie, nombre=None):
    """Todas las medidas de una columna numérica, juntas: centro, dispersión y forma."""
    s = pd.to_numeric(serie, errors="coerce")
    total = len(s)
    s = s.dropna()
    if len(s) == 0:
        return pd.Series({"n": 0}, name=nombre or serie.name)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    atipicos = int(((s > q3 + 1.5 * iqr) | (s < q1 - 1.5 * iqr)).sum())
    moda = s.mode() if s.nunique() <= max(20, 0.05 * len(s)) else pd.Series(dtype=float)   # la moda solo tiene sentido en valores que se repiten
    return pd.Series({
        "n": len(s), "% faltante": 100 * (1 - len(s) / total),
        "media": s.mean(), "mediana": s.median(), "moda": moda.iloc[0] if len(moda) else np.nan,
        "desvío": s.std(), "coef. de variación": s.std() / s.mean() if s.mean() else np.nan,
        "mínimo": s.min(), "Q1": q1, "Q3": q3, "máximo": s.max(), "IQR": iqr,
        "P90": s.quantile(0.90), "P99": s.quantile(0.99), "asimetría": s.skew(),
        "atípicos": atipicos, "% atípicos": 100 * atipicos / len(s),
    }, name=nombre or serie.name)

def reporte_inicial(df, nombre="tabla", mostrar=True):
    """PASO 1 · Describir. La primera descripción de cualquier tabla: forma, tipos, faltantes,
    duplicados, medidas de las columnas numéricas, resumen de las categóricas y rango de las fechas.
    Devuelve un diccionario con cada parte del reporte."""
    informe = {}
    informe["forma"] = pd.DataFrame({"filas": [len(df)], "columnas": [df.shape[1]], "filas duplicadas": [int(df.duplicated().sum())],
                                     "celdas faltantes": [int(df.isna().sum().sum())]}, index=[nombre])
    informe["columnas"] = pd.DataFrame({"tipo": df.dtypes.astype(str), "con dato": df.notna().sum(),
                                        "% faltante": 100 * df.isna().mean(), "valores distintos": df.nunique()})
    numericas = df.select_dtypes("number")
    if numericas.shape[1]:
        informe["numericas"] = pd.DataFrame({c: describir(numericas[c]) for c in numericas.columns}).T
    resto = df.select_dtypes(exclude=["number", "datetime"])
    if resto.shape[1]:
        filas = {}
        for c in resto.columns:
            frec = resto[c].value_counts(dropna=True)
            filas[c] = {"valores distintos": int(resto[c].nunique()), "más frecuente": frec.index[0] if len(frec) else None,
                        "% del más frecuente": 100 * frec.iloc[0] / resto[c].notna().sum() if len(frec) else np.nan,
                        "parece un identificador": bool(resto[c].nunique() == resto[c].notna().sum() and len(resto) > 1)}
        informe["categoricas"] = pd.DataFrame(filas).T
    fechas = df.select_dtypes("datetime")
    if fechas.shape[1]:
        informe["fechas"] = pd.DataFrame({c: {"primera": fechas[c].min(), "última": fechas[c].max(), "días cubiertos": (fechas[c].max() - fechas[c].min()).days} for c in fechas.columns}).T
    if mostrar:
        titulos = {"forma": "1 · FORMA", "columnas": "2 · TIPOS Y FALTANTES POR COLUMNA", "numericas": "3 · COLUMNAS NUMÉRICAS: centro, dispersión y forma",
                   "categoricas": "4 · COLUMNAS DE TEXTO Y CATEGORÍAS", "fechas": "5 · FECHAS"}
        for clave, titulo in titulos.items():
            if clave in informe:
                print(f"\n━━ {titulo}")
                display(informe[clave].round(2))
    return informe

def _periodos(df, periodo):
    p = sorted(df[periodo].dropna().unique())
    if len(p) != 2:
        raise ValueError(f"La columna «{periodo}» debe tener exactamente dos períodos y tiene {len(p)}: {p}")
    return p[0], p[1]

def cuantificar_cambio(df, valor, periodo, por=None, funcion="mean"):
    """PASO 2 · Cuantificar. Magnitud y dirección del cambio de `valor` entre los dos períodos de `periodo`.
    `funcion` puede ser "mean", "median", "sum" o "count". Con `por`, lo calcula dentro de cada grupo.
    Para una proporción, usá una columna de verdadero o falso y "mean"."""
    p1, p2 = _periodos(df, periodo)
    def resumen(d):
        a = d.loc[d[periodo] == p1, valor].dropna(); b = d.loc[d[periodo] == p2, valor].dropna()
        ma, mb = getattr(a, funcion)(), getattr(b, funcion)()
        return pd.Series({f"n {p1}": len(a), f"n {p2}": len(b), p1: ma, p2: mb, "diferencia": mb - ma,
                          "variación %": 100 * (mb / ma - 1) if ma else np.nan, "dirección": "↑" if mb > ma else "↓" if mb < ma else "="})
    if por is None:
        return resumen(df).to_frame("total").T
    return pd.DataFrame({g: resumen(d) for g, d in df.groupby(por)}).T

def descartar_impostores(df, valor, periodo, mezcla, universos=None, repeticiones=1000, semilla=0, mostrar=True):
    """PASO 3 · Descartar impostores. Un cambio puede ser un impostor por cuatro motivos:
       definición (¿cambia con otro universo?), mezcla (¿cambió la composición, no el comportamiento?),
       ruido (¿entra en la variación al azar?) y filtro (¿depende de los valores extremos?).
    `universos` es un diccionario {nombre: función que recibe df y devuelve una máscara} para probar definiciones alternativas."""
    p1, p2 = _periodos(df, periodo)
    a, b = df[df[periodo] == p1], df[df[periodo] == p2]
    cambio = b[valor].mean() - a[valor].mean()
    res = {"cambio": cambio}
    # a) definición
    if universos:
        filas = {}
        for nombre, regla in universos.items():
            d = df[regla(df)]
            filas[nombre] = {"n": len(d), "cambio de la media": d.loc[d[periodo] == p2, valor].mean() - d.loc[d[periodo] == p1, valor].mean()}
        res["definicion"] = pd.DataFrame(filas).T
    # b) mezcla: cambio = efecto dentro de cada grupo + efecto de la composición (la identidad es exacta)
    wa, wb = a[mezcla].value_counts(normalize=True), b[mezcla].value_counts(normalize=True)
    ma, mb = a.groupby(mezcla)[valor].mean(), b.groupby(mezcla)[valor].mean()
    idx = wa.index.union(wb.index)
    wa, wb, ma, mb = [x.reindex(idx).fillna(0) for x in (wa, wb, ma, mb)]
    dentro, composicion = wa * (mb - ma), (wb - wa) * mb
    res["mezcla"] = pd.DataFrame({f"peso {p1} %": 100 * wa, f"peso {p2} %": 100 * wb, f"media {p1}": ma, f"media {p2}": mb,
                                  "aporte dentro del grupo": dentro, "aporte de la composición": composicion})
    res["efecto_dentro"], res["efecto_composicion"] = dentro.sum(), composicion.sum()
    # c) ruido: remuestreo (bootstrap) de la diferencia de medias
    azar = np.random.default_rng(semilla)
    xa, xb = a[valor].dropna().to_numpy(float), b[valor].dropna().to_numpy(float)
    difs = np.array([azar.choice(xb, len(xb)).mean() - azar.choice(xa, len(xa)).mean() for _ in range(repeticiones)])
    res["ic95"] = tuple(np.percentile(difs, [2.5, 97.5]))
    # d) filtro: el cambio sin los valores extremos (regla del IQR sobre toda la tabla)
    todo = df[valor].dropna(); q1, q3 = todo.quantile([0.25, 0.75]); lim = q3 + 1.5 * (q3 - q1)
    sin = df[df[valor] <= lim]
    res["cambio_sin_atipicos"] = sin.loc[sin[periodo] == p2, valor].mean() - sin.loc[sin[periodo] == p1, valor].mean()
    res["lecturas"] = {
        "definición": "Mirá la tabla: si el cambio cambia de signo o de tamaño con otro universo, la definición manda." if universos else "No se probaron definiciones alternativas.",
        "mezcla": (f"La composición explica {100 * res['efecto_composicion'] / cambio:.0f} % del cambio y el comportamiento dentro de los grupos, {100 * res['efecto_dentro'] / cambio:.0f} %"
                   + (" (más del 100 % porque dentro de los grupos el cambio fue en sentido contrario)." if not 0 <= res['efecto_composicion'] / cambio <= 1 else ".")) if cambio else "No hay cambio para descomponer.",
        "ruido": "El intervalo incluye el 0: el cambio puede ser ruido." if res["ic95"][0] <= 0 <= res["ic95"][1] else "El intervalo no incluye el 0: el cambio es mayor que el ruido.",
        "filtro": f"Sin los valores extremos el cambio es {res['cambio_sin_atipicos']:+.2f} (con ellos, {cambio:+.2f}).",
    }
    if mostrar:
        print(f"Cambio de la media de «{valor}» entre {p1} y {p2}: {cambio:+.2f}")
        if universos:
            print("\n── a) DEFINICIÓN"); display(res["definicion"].round(2))
        print("\n── b) MEZCLA"); display(res["mezcla"].round(2))
        print(f"\n── c) RUIDO · intervalo de confianza del 95 % del cambio: [{res['ic95'][0]:+.2f}, {res['ic95'][1]:+.2f}]")
        print(f"── d) FILTRO · cambio sin valores extremos: {res['cambio_sin_atipicos']:+.2f}")
        print("\nLECTURA"); [print(f"  • {k}: {v}") for k, v in res["lecturas"].items()]
    return res

def cortes(df, valor, periodo, por_columnas, funcion="mean", minimo_n=30, mostrar=True):
    """¿El patrón sobrevive a los cortes? Repite el cambio dentro de cada grupo de cada columna de `por_columnas`
    y cuenta en cuántos grupos (con al menos `minimo_n` observaciones por período) va en la misma dirección que el total."""
    p1, p2 = _periodos(df, periodo)
    total = cuantificar_cambio(df, valor, periodo, None, funcion).iloc[0]
    signo = np.sign(total["diferencia"])
    filas = []
    for col in por_columnas:
        t = cuantificar_cambio(df, valor, periodo, col, funcion)
        for g, r in t.iterrows():
            n_min = int(min(r[f"n {p1}"], r[f"n {p2}"]))
            filas.append({"corte": col, "grupo": g, "n mínimo": n_min, "diferencia": r["diferencia"], "dirección": r["dirección"],
                          "n suficiente": n_min >= minimo_n, "misma dirección que el total": bool(np.sign(r["diferencia"]) == signo)})
    tabla = pd.DataFrame(filas)
    validos = tabla[tabla["n suficiente"]]
    fraccion = validos["misma dirección que el total"].mean() if len(validos) else np.nan
    veredicto = "sobrevive" if fraccion >= 0.75 else "sobrevive solo en parte" if fraccion >= 0.5 else "no sobrevive"
    resumen = f"El patrón {veredicto}: va en la misma dirección en {int(validos['misma dirección que el total'].sum())} de {len(validos)} grupos con datos suficientes."
    if mostrar:
        display(tabla.round(2)); print(resumen)
    return tabla, resumen

def procedimiento(df, valor, periodo, mezcla, cortes_por, universos=None, funcion="mean"):
    """Corre el procedimiento completo sobre una pregunta de cambio y deja para el equipo las preguntas finales."""
    p1, p2 = _periodos(df, periodo)
    print("═" * 8, "1 · DESCRIBIR: métrica, granularidad, período y universo")
    print(f"Métrica: {funcion} de «{valor}» · Períodos: {p1} y {p2} · Filas del universo: {len(df):,}")
    display(pd.DataFrame({p: describir(df.loc[df[periodo] == p, valor], p) for p in (p1, p2)}).round(2))
    print("\n" + "═" * 8, "2 · CUANTIFICAR: magnitud y dirección del cambio")
    display(cuantificar_cambio(df, valor, periodo, None, funcion).round(2))
    print("\n" + "═" * 8, "3 · DESCARTAR IMPOSTORES: definición, mezcla, ruido y filtro")
    impostores = descartar_impostores(df, valor, periodo, mezcla, universos)
    print("\n" + "═" * 8, "¿EL PATRÓN SOBREVIVE A LOS CORTES?")
    tabla, resumen = cortes(df, valor, periodo, cortes_por, funcion)
    print("\n" + "═" * 8, "4 Y 5 · LO QUE NO PUEDE HACER EL CÓDIGO")
    print("  4. Si el patrón sobrevive: formulá una hipótesis verificable. Si no: el patrón era un artefacto; reformulá.")
    print("  5. Recomendá, con los límites explícitos: qué decisión habilita y qué no podés afirmar.")
    return {"impostores": impostores, "cortes": tabla, "veredicto": resumen}


# ═══════════════════════════ EXTRAS ═══════════════════════════
import matplotlib
import matplotlib.pyplot as plt

_FECHA_ISO = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
_FECHA_LATINA = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}")

def _parece_fecha_texto(serie):
    """¿Es una columna de texto cuyos valores son fechas? (solo formatos AAAA-MM-DD y DD/MM/AAAA)"""
    if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_datetime64_any_dtype(serie) or pd.api.types.is_bool_dtype(serie):
        return False
    muestra = serie.dropna().astype(str).head(200)
    if len(muestra) == 0:
        return False
    if muestra.map(lambda v: bool(_FECHA_ISO.match(v) or _FECHA_LATINA.match(v))).mean() < 0.9:
        return False
    return _a_fecha(muestra).notna().mean() >= 0.9

def _a_fecha(serie):
    """Convierte texto a fecha. Si tiene formato DD/MM/AAAA, el día va primero."""
    latina = serie.dropna().astype(str).map(lambda v: bool(_FECHA_LATINA.match(v))).mean() >= 0.9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(serie, errors="coerce", dayfirst=bool(latina))

def cargar(ruta, **opciones):
    """Lee un CSV o un Excel, limpia los nombres de las columnas y convierte a fecha las que parecen fechas."""
    ruta = str(ruta)
    df = pd.read_excel(ruta, **opciones) if ruta.lower().endswith((".xlsx", ".xls")) else pd.read_csv(ruta, **opciones)
    df.columns = [str(c).strip() for c in df.columns]
    convertidas = [c for c in df.columns if _parece_fecha_texto(df[c])]
    for c in convertidas:
        df[c] = _a_fecha(df[c])
    print(f"Archivo cargado: {len(df):,} filas y {df.shape[1]} columnas." + (f" Convertidas a fecha: {', '.join(convertidas)}." if convertidas else ""))
    return df

def auditoria(df, umbral_faltantes=5.0, mostrar=True):
    """La primera descripción, convertida en una lista de decisiones. Detecta hallazgos de calidad y, para cada uno,
    dice qué hay que decidir. No corrige nada: la limpieza es una decisión de quien analiza."""
    filas = []
    def agregar(prioridad, columna, hallazgo, detalle, decision):
        filas.append({"prioridad": prioridad, "columna": columna, "hallazgo": hallazgo, "detalle": detalle, "decisión a tomar": decision})
    duplicadas = int(df.duplicated().sum())
    if duplicadas:
        agregar("alta", "(tabla)", "Filas duplicadas", f"{duplicadas:,} filas ({100 * duplicadas / len(df):.1f} %)", "¿Es un error de carga o son repeticiones válidas? Decidilo antes de contar.")
    clave_negativa = re.compile(r"precio|monto|importe|unidad|cantidad|edad|descuento|ticket|total", re.I)
    for c in df.columns:
        s = df[c]
        faltante = 100 * s.isna().mean()
        if faltante > 0:
            agregar("alta" if faltante > umbral_faltantes else "media", c, "Faltantes", f"{faltante:.1f} % ({int(s.isna().sum()):,} filas)", "¿Por qué falta y en qué grupos? Analizalo antes de borrar o reemplazar.")
        if s.nunique(dropna=True) <= 1:
            agregar("media", c, "Columna constante", "un solo valor" if s.notna().any() else "vacía", "No distingue casos: ¿se puede descartar?")
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna()
            q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
            atipicos = 100 * ((x > q3 + 1.5 * iqr) | (x < q1 - 1.5 * iqr)).mean() if iqr > 0 else 0.0
            if abs(x.skew()) > 1:
                agregar("media", c, "Distribución asimétrica", f"asimetría {x.skew():.2f}", "Usá mediana y percentiles junto con la media.")
            if atipicos > 5:
                agregar("media", c, "Muchos valores atípicos", f"{atipicos:.1f} % fuera de la regla del IQR", "Investigá su explicación antes de borrarlos.")
            if (x < 0).any() and clave_negativa.search(str(c)):
                agregar("alta", c, "Valores negativos", f"{int((x < 0).sum()):,} filas", "¿Es un dato válido (devolución, ajuste) o un error?")
            if pd.api.types.is_integer_dtype(s) and len(df) > 1 and s.nunique() == s.notna().sum():
                agregar("baja", c, "Parece un identificador", "todos los valores son distintos", "Úsala como clave, no para calcular ni agrupar.")
        else:
            if _parece_fecha_texto(s):
                agregar("alta", c, "Fecha guardada como texto", "los valores tienen formato de fecha", "Convertir a fecha para ordenar y agrupar por período.")
            elif len(df) > 1 and s.nunique() == s.notna().sum():
                agregar("baja", c, "Parece un identificador", "todos los valores son distintos", "Úsala como clave, no para agrupar.")
            elif s.nunique() <= 100:
                limpio = s.dropna().astype(str).str.strip().str.lower()
                if limpio.nunique() < s.nunique():
                    agregar("alta", c, "Categorías escritas distinto", f"{s.nunique()} valores que son {limpio.nunique()} al unificar mayúsculas y espacios", "Unificar antes de contar o agrupar.")
    orden = {"alta": 0, "media": 1, "baja": 2}
    tabla = pd.DataFrame(filas, columns=["prioridad", "columna", "hallazgo", "detalle", "decisión a tomar"])
    if len(tabla):
        tabla = tabla.sort_values(by=["prioridad", "columna"], key=lambda col: col.map(orden) if col.name == "prioridad" else col).reset_index(drop=True)
    if mostrar:
        print(f"Auditoría: {len(tabla)} hallazgos ({(tabla['prioridad'] == 'alta').sum() if len(tabla) else 0} de prioridad alta)")
        if len(tabla):
            display(tabla)
    return tabla

def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def _tabla_html(tabla, indice=True):
    return tabla.to_html(border=0, classes="tabla", index=indice, na_rep="", float_format=lambda v: f"{v:,.2f}", escape=True)

def _graficos(df, informe):
    graficos = {}
    colores = ("#7C5CF6", "#10D9A0", "#F59E0B")
    faltantes = (100 * df.isna().mean()).loc[lambda s: s > 0].sort_values()
    if len(faltantes):
        fig, ax = plt.subplots(figsize=(6, 0.45 * len(faltantes) + 1.2))
        ax.barh(faltantes.index.astype(str), faltantes.values, color=colores[2]); ax.set_xlabel("% de faltantes"); ax.set_title("Faltantes por columna")
        graficos["faltantes"] = _png(fig)
    numericas = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 1][:12]
    if numericas:
        columnas = min(3, len(numericas)); filas = int(np.ceil(len(numericas) / columnas))
        fig, ejes = plt.subplots(filas, columnas, figsize=(4.2 * columnas, 2.9 * filas), squeeze=False)
        for ax, c in zip(ejes.ravel(), numericas):
            x = df[c].dropna(); ax.hist(x, bins=min(40, max(5, x.nunique())), color=colores[0], alpha=0.9)
            ax.axvline(x.mean(), color=colores[2], lw=2); ax.axvline(x.median(), color=colores[1], lw=2); ax.set_title(str(c), fontsize=10)
        for ax in ejes.ravel()[len(numericas):]:
            ax.axis("off")
        fig.suptitle("Distribuciones · naranja: media · verde: mediana", y=1.0, fontsize=11); fig.tight_layout()
        graficos["numericas"] = _png(fig)
    categoricas = [c for c in df.select_dtypes(exclude=["number", "datetime"]).columns if 2 <= df[c].nunique() <= 30][:6]
    if categoricas:
        columnas = min(3, len(categoricas)); filas = int(np.ceil(len(categoricas) / columnas))
        fig, ejes = plt.subplots(filas, columnas, figsize=(4.2 * columnas, 2.9 * filas), squeeze=False)
        for ax, c in zip(ejes.ravel(), categoricas):
            frec = df[c].value_counts().head(8).sort_values(); ax.barh(frec.index.astype(str), frec.values, color=colores[0]); ax.set_title(str(c), fontsize=10)
        for ax in ejes.ravel()[len(categoricas):]:
            ax.axis("off")
        fig.tight_layout(); graficos["categoricas"] = _png(fig)
    if len(numericas) >= 2:
        corr = df[numericas].corr()
        fig, ax = plt.subplots(figsize=(0.7 * len(numericas) + 3, 0.6 * len(numericas) + 2.5))
        im = ax.imshow(corr, cmap="PuOr_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numericas)), numericas, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(len(numericas)), numericas, fontsize=8)
        for i in range(len(numericas)):
            for j in range(len(numericas)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Correlación (asociación lineal, no causalidad)"); plt.colorbar(im, shrink=0.8)
        graficos["correlacion"] = _png(fig)
    return graficos

_CSS = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#26262b;line-height:1.45}
h1{color:#5F22A7;margin-bottom:.2rem}h2{color:#5F22A7;border-bottom:2px solid #7C5CF6;padding-bottom:.25rem;margin-top:2.2rem}
.sub{color:#76768a;margin-top:0}.tabla{border-collapse:collapse;font-size:.85rem;margin:.6rem 0}.tabla th{background:#7C5CF6;color:#fff;padding:.35rem .6rem;text-align:left}
.tabla td{padding:.3rem .6rem;border-bottom:1px solid #e6e1f2}.tabla tr:nth-child(even) td{background:#f6f3fd}img{max-width:100%;margin:.6rem 0}
.nota{background:#f3effd;border-left:4px solid #7C5CF6;padding:.6rem .9rem;border-radius:4px;margin:.8rem 0}.alta{color:#c0392b;font-weight:600}
footer{margin-top:2.5rem;color:#76768a;font-size:.8rem}"""

def reporte_html(df, ruta, titulo="Reporte inicial", nombre="tabla", periodo=None, valor=None, mezcla=None, cortes_por=None, funcion="mean"):
    """Genera un reporte en un solo archivo HTML: la primera descripción, la auditoría de calidad y, si se indican
    `periodo` y `valor`, la comparación entre dos períodos con los pasos 2 y 3 y los cortes."""
    informe = reporte_inicial(df, nombre, mostrar=False)
    hallazgos = auditoria(df, mostrar=False)
    graficos = _graficos(df, informe)
    p = []
    p.append(f"<h1>{html.escape(titulo)}</h1><p class='sub'>{len(df):,} filas · {df.shape[1]} columnas · generado con el kit de análisis</p>")
    p.append("<h2>1 · Forma</h2>" + _tabla_html(informe["forma"]))
    p.append("<h2>2 · Auditoría de calidad: qué hay que decidir</h2>")
    if len(hallazgos):
        p.append("<p class='nota'>Cada hallazgo trae la decisión que le toca a quien analiza. El kit no corrige nada por su cuenta.</p>" + _tabla_html(hallazgos, indice=False))
    else:
        p.append("<p class='nota'>No se detectaron hallazgos con las reglas automáticas. Eso no reemplaza la revisión humana.</p>")
    p.append("<h2>3 · Tipos y faltantes por columna</h2>" + _tabla_html(informe["columnas"]) + (f"<img src='{graficos['faltantes']}'>" if "faltantes" in graficos else ""))
    if "numericas" in informe:
        p.append("<h2>4 · Columnas numéricas: centro, dispersión y forma</h2>" + _tabla_html(informe["numericas"]) + (f"<img src='{graficos['numericas']}'>" if "numericas" in graficos else ""))
    if "categoricas" in informe:
        p.append("<h2>5 · Columnas de texto y categorías</h2>" + _tabla_html(informe["categoricas"]) + (f"<img src='{graficos['categoricas']}'>" if "categoricas" in graficos else ""))
    if "fechas" in informe:
        p.append("<h2>6 · Fechas</h2>" + _tabla_html(informe["fechas"]))
    if "correlacion" in graficos:
        p.append("<h2>7 · Correlaciones</h2><p class='nota'>Una correlación alta no demuestra causalidad; una cercana a 0 solo descarta una relación en línea recta.</p>" + f"<img src='{graficos['correlacion']}'>")
    if periodo and valor:
        p.append(f"<h2>8 · Comparación de períodos: {html.escape(str(valor))} según {html.escape(str(periodo))}</h2>")
        p.append("<h3>Paso 2 · Cuantificar</h3>" + _tabla_html(cuantificar_cambio(df, valor, periodo, None, funcion), indice=False))
        if mezcla:
            imp = descartar_impostores(df, valor, periodo, mezcla, mostrar=False)
            p.append("<h3>Paso 3 · Descartar impostores</h3><ul>" + "".join(f"<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>" for k, v in imp["lecturas"].items()) + "</ul>")
            p.append("<p><b>Mezcla:</b></p>" + _tabla_html(imp["mezcla"]))
            p.append(f"<p><b>Ruido:</b> intervalo de confianza del 95 % del cambio: [{imp['ic95'][0]:+.2f}, {imp['ic95'][1]:+.2f}]</p>")
        if cortes_por:
            tabla, resumen = cortes(df, valor, periodo, cortes_por, funcion, mostrar=False)
            p.append("<h3>¿El patrón sobrevive a los cortes?</h3>" + f"<p class='nota'>{html.escape(resumen)}</p>" + _tabla_html(tabla, indice=False))
        p.append("<p class='nota'>Los pasos 4 y 5 no los hace el código: formulá una hipótesis verificable (o reformulá si el patrón no sobrevive) y recomendá con límites explícitos.</p>")
    p.append("<footer>Reporte generado con el kit de análisis. El kit calcula; el criterio decide.</footer>")
    documento = f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>{html.escape(titulo)}</title><style>{_CSS}</style></head><body>{''.join(p)}</body></html>"
    ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True); ruta.write_text(documento, encoding="utf-8")
    return str(ruta)

def exportar_excel(informe, ruta, hallazgos=None):
    """Guarda las tablas del reporte inicial (y la auditoría, si se pasa) en un libro de Excel, una hoja por tabla."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("Para exportar a Excel falta la librería openpyxl:  pip install openpyxl")
        return None
    ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta) as libro:
        if hallazgos is not None:
            hallazgos.to_excel(libro, sheet_name="auditoria", index=False)
        for nombre, tabla in informe.items():
            tabla.to_excel(libro, sheet_name=nombre[:31])
    return str(ruta)

def reporte_completo(origen, carpeta=".", nombre="reporte", titulo=None, periodo=None, valor=None, mezcla=None, cortes_por=None):
    """Todo en una llamada: carga el archivo (o usa la tabla que le pasen), hace la primera descripción y la auditoría,
    y guarda un reporte HTML y un libro de Excel. Devuelve un diccionario con las tablas y las rutas."""
    df = cargar(origen) if isinstance(origen, (str, Path)) else origen
    informe = reporte_inicial(df, nombre, mostrar=False)
    hallazgos = auditoria(df, mostrar=False)
    carpeta = Path(carpeta)
    ruta_html = reporte_html(df, carpeta / f"{nombre}.html", titulo or f"Reporte inicial · {nombre}", nombre, periodo, valor, mezcla, cortes_por)
    ruta_excel = exportar_excel(informe, carpeta / f"{nombre}.xlsx", hallazgos)
    excel_texto = ruta_excel or "no se generó (falta la librería openpyxl:  pip install openpyxl)"
    print(f"Listo · {len(hallazgos)} hallazgos de calidad ({(hallazgos['prioridad'] == 'alta').sum() if len(hallazgos) else 0} de prioridad alta)\n  Reporte HTML: {ruta_html}\n  Libro de Excel: {excel_texto}")
    return {"informe": informe, "hallazgos": hallazgos, "html": ruta_html, "excel": ruta_excel}
