# Scraper Argenprop — Propiedades en venta en Capital Federal

## Instalación

```bash
pip install requests beautifulsoup4 lxml pandas --break-system-packages
```

## Uso

El scraper vive en `src/real_estate/ingestion/scraper.py` y se ejecuta desde
el entry point `scripts/scrape.py`:

```bash
# Probar primero con pocas páginas para validar que anda bien en tu máquina
python scripts/scrape.py --max-paginas 5 --output data/raw/prueba.csv
```

Revisá `data/raw/prueba.csv` y, si los datos se ven bien, corré el scraping
completo (son ~100.000 avisos, así que puede tardar varias horas por los
delays entre requests):

```bash
python scripts/scrape.py --output data/raw/propiedades_argenprop.csv
```

Si se corta a mitad de camino (por ejemplo, cerrás la compu), volvé a
correr el mismo comando: el script no vuelve a bajar los avisos que ya
están en el CSV.

## Columnas del dataset

| Columna | Descripción |
|---|---|
| id | ID interno del aviso en Argenprop |
| link | URL del aviso |
| titulo | Título del aviso |
| descripcion | Copete/descripción corta |
| tipo_propiedad / idtipopropiedad | departamento, casa, ph, etc. (inferido de la URL) + código numérico interno del sitio |
| barrio / sub_barrio | Ubicación |
| precio / moneda | Precio (ya numérico) y si es USD o ARS |
| expensas | Expensas mensuales, como texto (si el aviso las publica) |
| superficie_cubierta / superficie_semicubierta / superficie_total | En m², como texto (ej. "90 m² cubie.") |
| ambientes | Cantidad, extraída de la URL del aviso (dato confiable) |
| dormitorios | Cantidad (atributo del aviso; puede faltar si el aviso no lo declaró) |
| banos / cocheras | Cantidad — **suelen quedar vacíos**: el listado solo muestra 2-3 características por tarjeta, no siempre las mismas, así que estos dos casi nunca aparecen ahí. Para completarlos habría que scrapear la página de detalle de cada aviso (ver más abajo). |
| antiguedad | Antigüedad en años, como texto ("17 años", "A estrenar", "En pozo", etc.) |
| fecha_scrape | Cuándo se bajó ese aviso |

### Sobre baños y cocheras

Estos dos campos van a tener muchos vacíos incluso con el scraper funcionando
perfectamente — no es un bug. Si para el modelo te importa tenerlos completos,
la opción es escribir un segundo script que visite `link` (la URL de cada
aviso) y extraiga esos datos de la ficha técnica completa. Esto implica una
request HTTP por cada aviso (además de las de listado), así que multiplica
bastante el tiempo total de scraping — te lo puedo armar si querés, una vez
que tengas el dataset base.

## Próximos pasos sugeridos para el modelo de ML

1. **Limpieza de tipos**: `precio` ya viene numérico. `superficie_cubierta`,
   `expensas`, etc. siguen viniendo como texto y hay que convertirlos a
   numérico (vas a encontrar formatos con puntos de miles, "m²", etc.)
2. **Normalizar moneda**: convertir todo a una sola moneda (USD suele ser
   más estable para este mercado) usando el tipo de cambio de la fecha
   de scrape.
3. **Manejo de NaN**: muchas columnas van a tener datos faltantes
   (expensas, antigüedad, baños) — decidír si imputar o dejarlos como
   señal ("el aviso no lo informó").
4. **Geolocalización**: si querés lat/long por barrio para el modelo,
   se puede geocodificar `barrio`/`sub_barrio` con un servicio como
   Nominatim (OpenStreetMap) en un paso aparte.
5. **Outliers**: hay avisos con precios claramente mal cargados (ej.
   precio en 0 o excesivamente alto) — conviene filtrarlos antes de
   entrenar.

Si querés, en la próxima conversación te ayudo con el script de limpieza
y el armado del modelo una vez que tengas el CSV.
