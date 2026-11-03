# Trabajo final · Data Engineer · ModaViva

**Moda online y tiendas** · Empresa ficticia, datos sintéticos.

## El caso

ModaViva vende por web, por app y en ocho tiendas. Cada sistema nombra los productos de otra manera y los precios cambian durante el año. Marketing quiere una vista única de ventas y recomendaciones personalizadas, pero el tráfico incluye bots.

## Qué tiene que resolver la plataforma

- Una vista única de ventas web y tienda por producto.
- Una base para recomendaciones y propensión de compra.
- Historia de precios.

## Restricciones del caso

- Tres formatos de SKU distintos.
- El clickstream incluye sesiones de bots y eventos duplicados.
- Correo de los clientes y comportamiento de navegación.

Todo lo que el caso **no declara** (volumen, disponibilidad, permisos, presupuesto) es una suposición: escribila y justificala.

## Las fuentes

Están en `fuentes/`: una base SQLite con tablas operacionales, archivos de terceros y eventos en JSON, más un `diccionario.md`.

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

## Qué tiene que recorrer tu trabajo

El trabajo integra todo el programa. Cada fila es parte de la evidencia que se evalúa. **Elegí la herramienta después de fijar el requisito.**

| Módulo | Qué se entrega |
| --- | --- |
| 1 · Arquitectura moderna | Un diagrama con fuentes, almacenamiento, procesamiento y consumo, y la justificación de la elección (warehouse, lake o lakehouse) con **una alternativa descartada**. |
| 2 · Modelado | Modelo conceptual, lógico y físico del dominio y un modelo dimensional: un hecho de ventas (web y tienda) con las dimensiones producto (con historia de precio), cliente y canal. Con su DDL. |
| 3 · PL/SQL | Un procedimiento (o un conjunto de sentencias SQL) que unifique los SKU y cargue las ventas de los tres orígenes, con manejo de errores y una medición de tiempos antes y después de optimizarlo. |
| 4 · Ingestión (NiFi) | El flujo para ingerir los pedidos web, los archivos diarios de tienda, el catálogo y el clickstream: procesadores, conexiones, manejo de errores y monitoreo. Se implementa en NiFi o, si no hay entorno, como diagrama más un script equivalente que cargue los datos entregados. |
| 5 · Orquestación (Airflow) | Un DAG con dependencias, calendario (diario para las ventas y por hora para el clickstream), reintentos e idempotencia, y la decisión sobre los datos que llegan tarde. |
| 6 · Governance | Al menos cinco reglas de calidad **medibles**, aplicadas a los datos entregados, con sus resultados; roles (dueño y responsable) y un contrato de datos. |
| 7 · MLOps | Un modelo de recomendación o de propensión de compra: ficha de versión, criterios de promoción, monitoreo y marcha atrás. |
| 8 · Plataformas | Las zonas de datos (cruda, curada, variables y consumo) con sus reglas de paso; si hay un modelo, la arquitectura para Machine Learning. |
| 9 · Seguridad y metadatos | Clasificación de los datos (correo de los clientes y su comportamiento de navegación), enmascaramiento y accesos por rol; un catálogo mínimo con el linaje de un indicador. |
| 10 · Proyecto | Una matriz de riesgos, un backlog priorizado y una defensa de 8 minutos. |

**Sobre el modelo (central):** Un modelo de recomendación o de propensión de compra: ficha de versión, criterios de promoción, monitoreo y marcha atrás.

## Qué se entrega

1. **Evidencia técnica**: el diagrama, el modelo, el DDL y una carga **ejecutable** sobre las fuentes entregadas (Python o SQL).
2. **Registro de decisiones**: cada decisión con su restricción, la alternativa descartada y el control (plantilla en `../plantillas.md`).
3. **Matriz de riesgos** y **backlog priorizado**.
4. **Defensa de 8 minutos.**

La rúbrica está en `../rubrica.md`. Si no hay entorno técnico, trabajá con tarjetas: cada tarjeta es una fuente, un proceso, un almacenamiento, un control o un consumidor; las conexiones se numeran y las fallas se simulan sacando una tarjeta.
