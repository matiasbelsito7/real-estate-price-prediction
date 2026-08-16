"""
Scraper de propiedades en venta en Capital Federal (Argenprop)
================================================================

Extrae avisos de venta de todo tipo de propiedad (departamentos, casas,
PH, etc.) publicados en Capital Federal, con variables pensadas para
entrenar un modelo de ML de predicción de precio.

Este módulo contiene la lógica de scraping (parseo de tarjetas, paginado
y guardado incremental). El punto de entrada con CLI es `scripts/scrape.py`.

v2: corrige el prefijo de clases CSS de los íconos de características,
que el sitio cambió de "icono-" a "basico1-icon-", y ahora usa además
los atributos de datos del propio link del aviso (id, dormitorios,
precio, moneda) en vez de depender solo de parsear texto, porque son
más confiables.

v3: agrega soporte para superar el cap de 100 páginas por búsqueda que
impone el sitio (HTTP 202 con cuerpo vacío a partir de la página 100 =
2000 avisos por búsqueda). Se scrapea segmentando por barrio y/o tipo
de propiedad, cada segmento con su propio paginado. Además maneja el
throttling anti-bot (202 también en páginas tempranas si el ritmo es
muy alto) con backoff exponencial + pausa larga, y persiste el progreso
por segmento para poder reanudar una corrida interrumpida.

NOTAS
-----
- Este scraper respeta al sitio: agrega delays entre requests y un
  User-Agent estándar. Si el sitio empieza a devolver errores o
  captchas de forma sostenida, es señal de que hay que bajar el ritmo
  (aumentar delay_min/delay_max) o parar.
- El sitio corta toda búsqueda en la página 100 (HTTP 202 vacío),
  aunque el widget de paginación muestre miles de páginas. Para
  acumular más de 2000 avisos hay que recorrer varios segmentos
  (barrios/tipos), cada uno con su propio cap de 100 páginas.
- Cada tarjeta de aviso en el listado solo muestra 2-3 características
  (de un conjunto variable: superficie cubierta, dormitorios,
  antigüedad, baños, ambientes). Por eso varias columnas van a tener
  NaN aunque el scraper esté funcionando bien: es una limitación del
  listado, no un bug. Para completar el 100% de esos datos habría que
  visitar la página de detalle de cada aviso (una request extra por
  aviso), lo cual multiplica mucho el tiempo total de scraping.
- 'cocheras' casi nunca aparece en la tarjeta del listado (solo en el
  detalle del aviso), así que esa columna va a quedar mayormente vacía.
- El precio y la moneda ahora salen de atributos de datos del propio
  aviso (ya vienen como número limpio), no de parsear texto.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from datetime import UTC, datetime
from typing import Any, cast

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.argenprop.com/inmuebles/venta/capital-federal"

# El sitio responde HTTP 202 con cuerpo vacío cuando (a) se supera el cap
# de páginas de una búsqueda o (b) detecta un ritmo de requests demasiado
# alto (throttling anti-bot). El cap es de 100 páginas x 20 avisos = 2000
# resultados por búsqueda; el widget de paginación muestra más páginas,
# pero el servidor nunca las entrega.
STATUS_BLOQUEO = 202
MAX_PAGINAS_SERVICIO = 100

# Slugs de los 54 barrios de Capital Federal, tal como los expone el
# filtro de ubicación del sitio. Se usan para segmentar la búsqueda y
# superar el cap de 2000 avisos por búsqueda.
BARRIOS_CABA = [
    "abasto",
    "agronomia",
    "almagro",
    "balvanera",
    "barracas",
    "barrio-norte",
    "barrio-santa-rita",
    "belgrano",
    "boca",
    "boedo",
    "caballito",
    "centro",
    "chacarita",
    "coghlan",
    "colegiales",
    "congreso",
    "constitucion",
    "flores",
    "floresta",
    "liniers",
    "mataderos",
    "monserrat",
    "monte-castro",
    "nunez",
    "once",
    "palermo",
    "parque-avellaneda",
    "parque-centenario",
    "parque-chacabuco",
    "parque-chas",
    "parque-patricios",
    "paternal",
    "pompeya",
    "puerto-madero",
    "recoleta",
    "retiro",
    "saavedra",
    "san-cristobal",
    "san-nicolas",
    "san-telmo",
    "velez-sarsfield",
    "versalles",
    "villa-crespo",
    "villa-del-parque",
    "villa-devoto",
    "villa-general-mitre",
    "villa-lugano",
    "villa-luro",
    "villa-ortuzar",
    "villa-pueyrredon",
    "villa-real",
    "villa-riachuelo",
    "villa-soldati",
    "villa-urquiza",
]

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

COLUMNS = [
    "id",
    "link",
    "titulo",
    "descripcion",
    "tipo_propiedad",
    "idtipopropiedad",
    "barrio",
    "sub_barrio",
    "precio",
    "moneda",
    "expensas",
    "superficie_cubierta",
    "superficie_semicubierta",
    "superficie_total",
    "ambientes",
    "dormitorios",
    "banos",
    "cocheras",
    "antiguedad",
    "fecha_scrape",
]

# Tipos de propiedad que aparecen en la URL del aviso, ej:
# https://www.argenprop.com/departamento-en-venta-en-palermo-3-ambientes--123
TIPOS_PROPIEDAD = [
    "departamento",
    "casa",
    "ph",
    "terreno",
    "local",
    "campo",
    "cochera",
    "fondo-de-comercio",
    "galpon",
    "hotel",
    "oficina",
    "quinta",
    "duplex",
    "triplex",
    "penthouse",
    "piso",
    "semipiso",
    "loft",
]

# Mapeo de sufijo de ícono (basico1-icon-XXX) -> columna del dataset.
# Cada tarjeta del listado muestra solo un subconjunto de estos, variable
# aviso a aviso, así que se recorren todos los que aparezcan.
ICONO_A_COLUMNA = {
    "superficie_cubierta": "superficie_cubierta",
    "superficie_semicubierta": "superficie_semicubierta",
    "superficie_total": "superficie_total",
    "cantidad_dormitorios": "dormitorios",
    "cantidad_ambientes": "ambientes",
    "cantidad_banos": "banos",
    "cantidad_cocheras": "cocheras",
    "antiguedad": "antiguedad",
}

RE_AMBIENTES_EN_URL = re.compile(r"-(\d+)-ambientes?--\d+$")


def construir_url_segmento(tipo: str | None = None, barrio: str | None = None) -> str:
    """Construye la URL base de un segmento de búsqueda.

    Sin filtros usa la búsqueda completa de Capital Federal. Con `tipo`
    (p. ej. "departamentos" o "casas") restringe el tipo de propiedad, y
    con `barrio` (slug, p. ej. "palermo") restringe a un barrio. Cada
    segmento tiene su propio paginado y su propio cap de 100 páginas.
    """
    url = BASE_URL
    if tipo:
        url = f"https://www.argenprop.com/inmuebles/{tipo}-en-venta/capital-federal"
    if barrio:
        url = f"{url}/{barrio}"
    return url


def construir_url_pagina(pagina: int, base_url: str = BASE_URL) -> str:
    if pagina <= 1:
        return base_url
    return f"{base_url}?pagina-{pagina}"


def texto_o_none(elemento: Any) -> str | None:
    return cast(str, elemento.text).strip() if elemento else None


def detectar_tipo_propiedad(href: str | None) -> str | None:
    if not href:
        return None
    href_lower = href.lower()
    for tipo in TIPOS_PROPIEDAD:
        if href_lower.startswith(f"/{tipo}-en-venta"):
            return tipo
    return None


def extraer_ambientes_de_url(href: str | None) -> str | None:
    if not href:
        return None
    match = RE_AMBIENTES_EN_URL.search(href)
    return match.group(1) if match else None


def extraer_features_de_tarjeta(listing: Any) -> dict[str, str]:
    """Recorre los <li> de card__main-features (superficie, dormitorios,
    antigüedad, etc.) y devuelve un dict {columna: valor}. La cantidad y
    el tipo de features que aparecen varían aviso a aviso."""
    resultado: dict[str, str] = {}
    for li in listing.select("ul.card__main-features li"):
        icono = li.find("i")
        span = li.find("span")
        if not icono or not span:
            continue
        clases = icono.get("class", [])
        clase_icono = next((c for c in clases if c.startswith("basico1-icon-")), None)
        if not clase_icono:
            continue
        sufijo = clase_icono.replace("basico1-icon-", "")
        columna = ICONO_A_COLUMNA.get(sufijo)
        if columna:
            resultado[columna] = cast(str, span.text).strip()
    return resultado


def parsear_listing(listing: Any) -> dict[str, Any] | None:
    link_tag = listing.find("a")
    if not link_tag:
        return None

    id_ = link_tag.get("data-item-card")
    href = link_tag.get("href", "")
    if not id_:
        return None

    link = "https://www.argenprop.com" + href if href.startswith("/") else href

    titulo = texto_o_none(listing.find(class_="card__title"))
    descripcion = texto_o_none(listing.find(class_="card__info"))
    ubicacion_texto = texto_o_none(listing.find(class_="card__title--primary"))

    barrio: str | None = None
    sub_barrio: str | None = None
    if ubicacion_texto:
        # Suele venir como "Departamento en Venta en <Zona>, <Barrio>"
        partes = re.split(r"\ben venta en\b", ubicacion_texto, flags=re.IGNORECASE)
        if len(partes) > 1:
            ubicacion = [p.strip() for p in partes[-1].strip().split(",")]
            if len(ubicacion) >= 2:
                sub_barrio, barrio = ubicacion[0], ubicacion[-1]
            elif ubicacion:
                barrio = ubicacion[0]
            if barrio and barrio.strip().lower() == "capital federal":
                barrio, sub_barrio = sub_barrio, None

    # Precio y moneda: uso los atributos de datos del propio link, que ya
    # vienen como número limpio (más confiable que parsear el texto de la
    # tarjeta, que mezcla precio + expensas en el mismo bloque).
    precio = link_tag.get("montooperacion") or link_tag.get("montonormalizado")
    moneda = texto_o_none(listing.find(class_="card__currency"))
    if not moneda:
        # Fallback por si el ícono de moneda no está: 1=ARS, 2=USD en Argenprop
        idmoneda = link_tag.get("idmoneda")
        moneda = {"1": "ARS", "2": "USD"}.get(idmoneda)

    expensas = texto_o_none(listing.find(class_="card__expenses"))

    idtipopropiedad = link_tag.get("idtipopropiedad")
    tipo_propiedad = detectar_tipo_propiedad(href)

    features = extraer_features_de_tarjeta(listing)

    # Dormitorios: prioridad al atributo del link (más estable), y si no
    # está, al que se haya podido leer de la tarjeta.
    dormitorios = link_tag.get("dormitorios") or features.get("dormitorios")
    if dormitorios == "":
        dormitorios = None

    # Ambientes: el atributo "ambientes" del link viene casi siempre vacío,
    # así que priorizo el que aparece en la URL del aviso (ahí sí es
    # consistente) y si no, el de la tarjeta.
    ambientes = extraer_ambientes_de_url(href) or features.get("ambientes")

    return {
        "id": id_,
        "link": link,
        "titulo": titulo,
        "descripcion": descripcion,
        "tipo_propiedad": tipo_propiedad,
        "idtipopropiedad": idtipopropiedad,
        "barrio": barrio,
        "sub_barrio": sub_barrio,
        "precio": precio,
        "moneda": moneda,
        "expensas": expensas,
        "superficie_cubierta": features.get("superficie_cubierta"),
        "superficie_semicubierta": features.get("superficie_semicubierta"),
        "superficie_total": features.get("superficie_total"),
        "ambientes": ambientes,
        "dormitorios": dormitorios,
        "banos": features.get("banos"),
        "cocheras": features.get("cocheras"),
        "antiguedad": features.get("antiguedad"),
        "fecha_scrape": datetime.now(UTC).isoformat(),
    }


def cargar_ids_existentes(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    ids: set[str] = set()
    with open(output_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            if fila.get("id"):
                ids.add(fila["id"])
    return ids


def asegurar_encabezado(output_path: str) -> None:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def guardar_filas(output_path: str, filas: list[dict[str, Any]]) -> None:
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        for fila in filas:
            writer.writerow(fila)


# ---------------------------------------------------------------------------
# Progreso por segmento (para reanudar corridas interrumpidas)
# ---------------------------------------------------------------------------


def cargar_progreso(archivo_progreso: str) -> dict[str, Any]:
    """Lee el archivo de progreso JSON. Devuelve {} si no existe o está corrupto."""
    if not os.path.exists(archivo_progreso):
        return {}
    try:
        with open(archivo_progreso, encoding="utf-8") as f:
            contenido = json.load(f)
        return contenido if isinstance(contenido, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def guardar_progreso(archivo_progreso: str, estado: dict[str, Any]) -> None:
    """Escribe el estado de progreso de todos los segmentos."""
    directorio = os.path.dirname(os.path.abspath(archivo_progreso))
    os.makedirs(directorio, exist_ok=True)
    with open(archivo_progreso, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)


def pagina_de_reanudacion(archivo_progreso: str, nombre_segmento: str) -> int | None:
    """Página desde la que reanudar un segmento, o None si no hay progreso.

    Devuelve None si el segmento no existe en el progreso o ya quedó
    marcado como completo. Si quedó a mitad, devuelve la última página
    procesada con éxito + 1.
    """
    segmento = cargar_progreso(archivo_progreso).get(nombre_segmento)
    if not segmento or segmento.get("completo"):
        return None
    ultima_pagina = segmento.get("pagina")
    if not isinstance(ultima_pagina, int):
        return None
    return ultima_pagina + 1


def scrapear(
    output_path: str,
    max_paginas: int | None,
    pagina_inicio: int,
    delay_min: float,
    delay_max: float,
    html_debug: str | None = None,
    base_url: str = BASE_URL,
    nombre_segmento: str = "global",
    archivo_progreso: str | None = None,
    reintentos_202: int = 5,
    backoff_202_inicial: float = 15.0,
    pausa_bloqueo: float = 300.0,
) -> None:
    asegurar_encabezado(output_path)
    ids_existentes = cargar_ids_existentes(output_path)
    print(f"IDs ya guardados previamente: {len(ids_existentes)}")

    estado_progreso: dict[str, Any] = {}
    if archivo_progreso:
        estado_progreso = cargar_progreso(archivo_progreso)
        if (estado_progreso.get(nombre_segmento) or {}).get("completo"):
            print(f"Segmento '{nombre_segmento}' ya quedó completo en el progreso. Lo salteo.")
            return
        reanudar_en = pagina_de_reanudacion(archivo_progreso, nombre_segmento)
        if reanudar_en is not None:
            pagina_inicio = reanudar_en
            print(f"Reanudo el segmento '{nombre_segmento}' desde la página {pagina_inicio}.")

    session = requests.Session()
    session.headers.update(HEADERS)

    pagina = pagina_inicio
    total_nuevos = 0
    ultima_pagina_ok = pagina - 1
    paginas_vacias_seguidas = 0
    paginas_sin_features_seguidas = 0
    reintentos_202_actual = 0
    pausas_largas = 0
    segmento_completo = False

    while True:
        if max_paginas is not None and (pagina - pagina_inicio) >= max_paginas:
            print(f"Llegué al límite de {max_paginas} páginas. Corto acá.")
            segmento_completo = True
            break

        url = construir_url_pagina(pagina, base_url)
        try:
            resp = session.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"Error de red en página {pagina}: {e}. Reintento en 10s...")
            time.sleep(10)
            continue

        if resp.status_code == 404:
            print(f"Página {pagina} devolvió 404: asumo que llegué al final del listado.")
            segmento_completo = True
            break

        if resp.status_code == STATUS_BLOQUEO:
            if pagina >= MAX_PAGINAS_SERVICIO:
                print(
                    f"Página {pagina}: el sitio corta la paginación en "
                    f"{MAX_PAGINAS_SERVICIO} páginas (HTTP 202 vacío). "
                    "Fin del segmento."
                )
                segmento_completo = True
                break
            reintentos_202_actual += 1
            if reintentos_202_actual <= reintentos_202:
                espera = min(backoff_202_inicial * (2 ** (reintentos_202_actual - 1)), 120)
                print(
                    f"Página {pagina}: el sitio bloqueó el request (HTTP 202). "
                    f"Reintento {reintentos_202_actual}/{reintentos_202} "
                    f"en {espera:.0f}s..."
                )
                time.sleep(espera)
                continue
            # Bloqueo sostenido: pausa larga y una nueva tanda de reintentos.
            reintentos_202_actual = 0
            pausas_largas += 1
            if pausas_largas > 2:
                print(
                    f"Página {pagina}: bloqueo persistente del sitio tras "
                    f"{pausas_largas} pausas largas. Guardo el progreso y corto "
                    "el segmento; se reanuda con el mismo comando."
                )
                segmento_completo = False
                break
            print(
                f"Página {pagina}: bloqueo sostenido del sitio. "
                f"Pausa larga de {pausa_bloqueo:.0f}s antes de reintentar..."
            )
            time.sleep(pausa_bloqueo)
            continue

        if resp.status_code != 200:
            # Otros errores (500, 429, etc.): reintento acotado también, con
            # el mismo backoff, para no quedar en loop infinito.
            reintentos_202_actual += 1
            if reintentos_202_actual <= reintentos_202:
                espera = min(backoff_202_inicial * (2 ** (reintentos_202_actual - 1)), 120)
                print(
                    f"Página {pagina}: status {resp.status_code}. "
                    f"Reintento {reintentos_202_actual}/{reintentos_202} "
                    f"en {espera:.0f}s..."
                )
                time.sleep(espera)
                continue
            print(f"Página {pagina}: status {resp.status_code} persistente. Corto el segmento.")
            segmento_completo = False
            break

        reintentos_202_actual = 0

        if html_debug:
            with open(html_debug, "w", encoding="utf-8") as f:
                f.write(resp.text)

        soup = BeautifulSoup(resp.text, "lxml")
        listings = soup.find_all("div", class_=lambda x: x == "listing__item" if x else False)

        if not listings:
            paginas_vacias_seguidas += 1
            print(f"Página {pagina}: sin avisos.")
            if paginas_vacias_seguidas >= 2:
                print("Dos páginas vacías seguidas: termino el scraping.")
                segmento_completo = True
                break
            pagina += 1
            time.sleep(random.uniform(delay_min, delay_max))
            continue

        paginas_vacias_seguidas = 0
        filas_pagina: list[dict[str, Any]] = []
        for listing in listings:
            fila = parsear_listing(listing)
            if not fila:
                continue
            if fila["id"] in ids_existentes:
                continue
            ids_existentes.add(fila["id"])
            filas_pagina.append(fila)

        if filas_pagina:
            guardar_filas(output_path, filas_pagina)
            total_nuevos += len(filas_pagina)

        # Alerta temprana si el sitio volvió a cambiar de estructura: si
        # ninguna fila de la página trajo superficie_cubierta NI
        # dormitorios, probablemente cambiaron las clases de nuevo.
        alguna_con_datos = any(
            f.get("superficie_cubierta") or f.get("dormitorios") for f in filas_pagina
        )
        if filas_pagina and not alguna_con_datos:
            paginas_sin_features_seguidas += 1
            if paginas_sin_features_seguidas >= 3:
                print(
                    "\n⚠️  ADVERTENCIA: las últimas 3 páginas no trajeron ni "
                    "superficie_cubierta ni dormitorios en ningún aviso. Es "
                    "probable que el sitio haya cambiado la estructura del "
                    "HTML de nuevo. Corré con --html-debug pagina.html para "
                    "guardar el HTML crudo e inspeccionarlo.\n"
                )
        else:
            paginas_sin_features_seguidas = 0

        print(
            f"Página {pagina}: {len(listings)} avisos encontrados, {len(filas_pagina)} nuevos guardados. Total nuevos: {total_nuevos}"
        )

        ultima_pagina_ok = pagina
        pagina += 1
        time.sleep(random.uniform(delay_min, delay_max))

        if archivo_progreso:
            estado_progreso[nombre_segmento] = {
                "pagina": ultima_pagina_ok,
                "completo": False,
            }
            guardar_progreso(archivo_progreso, estado_progreso)

    if archivo_progreso:
        estado_progreso[nombre_segmento] = {
            "pagina": ultima_pagina_ok,
            "completo": segmento_completo,
        }
        guardar_progreso(archivo_progreso, estado_progreso)
        estado = "completo" if segmento_completo else "incompleto"
        print(
            f"Progreso del segmento '{nombre_segmento}' guardado en "
            f"'{archivo_progreso}' ({estado}, hasta página {ultima_pagina_ok})."
        )

    print(f"\nListo. Se agregaron {total_nuevos} avisos nuevos a '{output_path}'.")
