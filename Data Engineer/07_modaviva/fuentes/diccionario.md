# Fuentes · ModaViva

ModaViva es una empresa **ficticia**. Los datos son sintéticos, salen de un generador con semilla fija y ningún resultado representa una situación real de ARCA.

## Inventario

| Fuente | Qué es | Filas |
| --- | --- | --- |
| `operacional.db` · `clientes` | Tabla de la base operacional | 4.000 |
| `operacional.db` · `pedidos_web` | Tabla de la base operacional | 9.000 |
| `operacional.db` · `items_pedido` | Tabla de la base operacional | 13.197 |
| `archivos_externos/catalogo.csv` | Un archivo que llega de un tercero | 1.338 |
| `archivos_externos/equivalencias_sku.csv` | Un archivo que llega de un tercero | 994 |
| `archivos_externos/devoluciones.csv` | Un archivo que llega de un tercero | 765 |
| `archivos_externos/ventas_tienda…` | 30 archivos que llegan de un tercero | 3.678 |
| `eventos/clickstream.jsonl` | Eventos en formato JSON, uno por línea | 86.440 |

## Columnas

### `operacional.db` · `clientes`

`id_cliente`, `email`, `ciudad`

### `operacional.db` · `pedidos_web`

`id_pedido`, `id_cliente`, `fecha`, `canal`, `estado`

### `operacional.db` · `items_pedido`

`id_pedido`, `sku_web`, `cantidad`, `precio_unitario`

### `archivos_externos/catalogo.csv…`

`sku_maestro`, `nombre`, `categoria`, `color`, `talle`, `precio_lista`, `vigente_desde`

### `archivos_externos/equivalencias_sku.csv…`

`sku_web`, `sku_maestro`

### `archivos_externos/devoluciones.csv…`

`id_pedido`, `sku_web`, `motivo`, `fecha`

### `archivos_externos/ventas_tienda…`

`ticket`, `sucursal`, `sku_tienda`, `unidades`, `precio`

### `eventos/clickstream.jsonl`

`session_id`, `cliente_id`, `ts`, `evento`, `sku_web`

Los archivos y los eventos pueden traer **cambios de esquema, duplicados y datos tardíos**: parte del trabajo es descubrirlos, decidir qué hacer con cada uno y dejarlo escrito.