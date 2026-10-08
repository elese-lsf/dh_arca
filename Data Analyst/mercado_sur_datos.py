"""Datos ficticios de Mercado Sur, generados desde código (sin archivos que subir).

    from mercado_sur_datos import generar_mercado_sur
    t = generar_mercado_sur()      # t["clientes"], t["productos"], t["ventas"]

Mismo código, misma semilla, mismas tablas. Ningún dato representa una situación real.
"""
import numpy as np
import pandas as pd

CATEGORIAS = {  # categoría: (precio mínimo, precio máximo, peso en las ventas)
    "Electrónica": (150, 900, 0.15),
    "Hogar": (30, 300, 0.20),
    "Indumentaria": (20, 150, 0.20),
    "Deportes": (25, 250, 0.15),
    "Alimentos": (3, 40, 0.30),
}
REGIONES = ["Centro", "Norte", "Sur", "Litoral", "Cuyo"]
PESOS_REGION = [0.30, 0.20, 0.18, 0.17, 0.15]


def generar_mercado_sur(semilla=2025, n_clientes=1500, n_ops=12000):
    rng = np.random.default_rng(semilla)

    # ---- productos: 8 por categoría
    filas, n = [], 1
    for cat, (lo, hi, _) in CATEGORIAS.items():
        for _ in range(8):
            filas.append({"id_producto": f"P{n:03d}", "categoria": cat, "nombre": f"{cat} {n:02d}",
                          "precio_lista": round(float(rng.uniform(lo, hi)), 2)})
            n += 1
    productos = pd.DataFrame(filas)

    # ---- clientes
    clientes = pd.DataFrame({
        "id_cliente": [f"C{i:04d}" for i in range(1, n_clientes + 1)],
        "region": rng.choice(REGIONES, size=n_clientes, p=PESOS_REGION),
        "fecha_alta": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n_clientes), unit="D"),
    })
    clientes["fecha_alta"] = clientes["fecha_alta"].dt.strftime("%Y-%m-%d")

    # ---- ventas: qué pasó (fecha, quién, por qué canal, qué producto)
    fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_ops), unit="D")
    ventas = pd.DataFrame({
        "id_venta": [f"V{i:06d}" for i in range(1, n_ops + 1)],
        "fecha": fechas,
        "id_cliente": rng.choice(clientes["id_cliente"], size=n_ops),
        "canal": rng.choice(["web", "tienda"], size=n_ops, p=[0.55, 0.45]),
    })
    pesos_cat = np.array([v[2] for v in CATEGORIAS.values()])
    cats = np.array(list(CATEGORIAS.keys()))[rng.choice(len(CATEGORIAS), size=n_ops, p=pesos_cat / pesos_cat.sum())]
    ventas["id_producto"] = [rng.choice(productos.loc[productos["categoria"] == c, "id_producto"]) for c in cats]
    ventas = ventas.merge(productos[["id_producto", "categoria", "precio_lista"]], on="id_producto", how="left")
    ventas = ventas.merge(clientes[["id_cliente", "region"]], on="id_cliente", how="left")
    ventas["unidades"] = np.where(ventas["categoria"] == "Alimentos", rng.integers(1, 9, n_ops), rng.integers(1, 5, n_ops))
    ventas["precio_unitario"] = (ventas["precio_lista"] * rng.uniform(0.97, 1.05, n_ops)).round(2)
    ventas["descuento_pct"] = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n_ops, p=[0.45, 0.15, 0.20, 0.12, 0.08])

    # ---- el patrón escondido del caso: en el 2.º semestre, un segmento cae y devuelve mucho
    s2 = ventas["fecha"].dt.month > 6
    seg = (ventas["canal"] == "web") & (ventas["region"] == "Litoral") & (ventas["categoria"] == "Electrónica") & s2
    seg_indum = (ventas["canal"] == "web") & (ventas["categoria"] == "Indumentaria") & s2
    p_comp, p_dev = np.full(n_ops, 0.88), np.full(n_ops, 0.07)
    p_can = np.full(n_ops, 0.05)
    p_comp[seg], p_dev[seg], p_can[seg] = 0.64, 0.30, 0.06
    p_comp[seg_indum], p_dev[seg_indum], p_can[seg_indum] = 0.81, 0.13, 0.06
    u = rng.random(n_ops)
    ventas["estado"] = np.where(u < p_comp, "completada", np.where(u < p_comp + p_dev, "devuelta", "cancelada"))
    ventas = ventas.loc[~(seg & (rng.random(n_ops) < 0.38))].copy()          # además, el segmento vende menos

    # ---- satisfacción (con faltantes que hay que analizar, no reemplazar)
    n_v = len(ventas)
    sat = rng.choice([1, 2, 3, 4, 5], size=n_v, p=[0.05, 0.08, 0.17, 0.35, 0.35]).astype(float)
    dev = (ventas["estado"] == "devuelta").to_numpy()
    can = (ventas["estado"] == "cancelada").to_numpy()
    sat[dev] = rng.choice([1, 2, 3], size=dev.sum(), p=[0.5, 0.3, 0.2])
    sat[rng.random(n_v) < np.where(can, 0.80, np.where(dev, 0.25, 0.12))] = np.nan
    ventas["satisfaccion"] = sat

    ventas = ventas.sort_values("fecha").reset_index(drop=True)
    ventas["id_venta"] = [f"V{i:06d}" for i in range(1, len(ventas) + 1)]
    ventas["fecha"] = ventas["fecha"].dt.strftime("%Y-%m-%d")
    ventas = ventas[["id_venta", "fecha", "id_cliente", "id_producto", "canal", "unidades",
                     "precio_unitario", "descuento_pct", "estado", "satisfaccion"]]
    return {"clientes": clientes[["id_cliente", "region", "fecha_alta"]],
            "productos": productos[["id_producto", "categoria", "nombre", "precio_lista"]],
            "ventas": ventas}
