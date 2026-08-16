"""Tests unitarios de `real_estate.ingestion.scraper`.

El HTML sintético replica la estructura real de las tarjetas de Argenprop
(validada contra el CSV producido: 2.005 filas, precio/moneda/
idtipopropiedad sin nulos). El parser lee atributos literales del link
(montooperacion, idmoneda, ...) y `data-item-card` para el id.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from real_estate.ingestion.scraper import (
    BASE_URL,
    COLUMNS,
    asegurar_encabezado,
    cargar_ids_existentes,
    construir_url_pagina,
    detectar_tipo_propiedad,
    extraer_ambientes_de_url,
    extraer_features_de_tarjeta,
    guardar_filas,
    parsear_listing,
)

HTML_TARJETA_COMPLETA = """
<div class="listing__item">
  <a data-item-card="999"
     montooperacion="250000"
     montonormalizado="250000"
     idmoneda="2"
     dormitorios="2"
     idtipopropiedad="1"
     href="/departamento-en-venta-en-palermo-3-ambientes--999">
     <div class="card__title--primary">Departamento en Venta en Palermo Chico, Palermo</div>
     <div class="card__currency">USD</div>
     <ul class="card__main-features">
       <li><i class="basico1-icon-cantidad_dormitorios"></i><span>2</span></li>
       <li><i class="basico1-icon-superficie_cubierta"></i><span>90 m2 cubie.</span></li>
       <li><i class="basico1-icon-antiguedad"></i><span>17 años</span></li>
     </ul>
  </a>
</div>
"""

HTML_TARJETA_MINIMA = """
<div class="listing__item">
  <a data-item-card="1000"
     montooperacion="150000"
     idmoneda="1"
     href="/casa-en-venta-en-palermo-4-ambientes--1000">
     <div class="card__title--primary">Casa en Venta en Palermo, Capital Federal</div>
  </a>
</div>
"""


def _listing_del_html(html: str) -> object:
    soup = BeautifulSoup(html, "lxml")
    return soup.find_all("div", class_="listing__item")[0]


class TestConstruirUrlPagina:
    def test_pagina_uno_es_base(self) -> None:
        assert construir_url_pagina(1) == BASE_URL

    def test_pagina_mayor_agrega_query(self) -> None:
        assert construir_url_pagina(3) == f"{BASE_URL}?pagina-3"

    def test_pagina_cero_tratada_como_primera(self) -> None:
        assert construir_url_pagina(0) == BASE_URL


class TestDetectarTipoPropiedad:
    def test_departamento(self) -> None:
        assert detectar_tipo_propiedad("/departamento-en-venta-en-palermo--1") == "departamento"

    def test_casa(self) -> None:
        assert detectar_tipo_propiedad("/casa-en-venta-en-belgrano--2") == "casa"

    def test_sin_href(self) -> None:
        assert detectar_tipo_propiedad(None) is None

    def test_href_desconocido(self) -> None:
        assert detectar_tipo_propiedad("/otra-cosa-en-venta-en-x--3") is None


class TestExtraerAmbientesDeUrl:
    def test_ambientes_de_url(self) -> None:
        assert extraer_ambientes_de_url("/departamento-en-venta-en-palermo-3-ambientes--999") == "3"

    def test_sin_href(self) -> None:
        assert extraer_ambientes_de_url(None) is None

    def test_url_sin_ambientes(self) -> None:
        assert extraer_ambientes_de_url("/terreno-en-venta-en-x--5") is None


class TestExtraerFeaturesDeTarjeta:
    def test_extrae_features_conocidos(self) -> None:
        listing = _listing_del_html(HTML_TARJETA_COMPLETA)
        features = extraer_features_de_tarjeta(listing)
        assert features == {
            "dormitorios": "2",
            "superficie_cubierta": "90 m2 cubie.",
            "antiguedad": "17 años",
        }

    def test_ignora_iconos_desconocidos(self) -> None:
        html = """
        <div class="listing__item">
          <ul class="card__main-features">
            <li><i class="basico1-icon-algo_nuevo"></i><span>X</span></li>
          </ul>
        </div>
        """
        listing = _listing_del_html(html)
        assert extraer_features_de_tarjeta(listing) == {}


class TestParsearListing:
    def test_tarjeta_completa(self) -> None:
        listing = _listing_del_html(HTML_TARJETA_COMPLETA)
        resultado = parsear_listing(listing)

        assert resultado is not None
        assert resultado["id"] == "999"
        assert resultado["tipo_propiedad"] == "departamento"
        assert resultado["idtipopropiedad"] == "1"
        assert resultado["precio"] == "250000"
        assert resultado["moneda"] == "USD"
        assert resultado["superficie_cubierta"] == "90 m2 cubie."
        assert resultado["dormitorios"] == "2"
        assert resultado["ambientes"] == "3"
        assert resultado["antiguedad"] == "17 años"
        assert resultado["barrio"] == "Palermo"
        assert resultado["sub_barrio"] == "Palermo Chico"
        assert resultado["link"] == (
            "https://www.argenprop.com/departamento-en-venta-en-palermo-3-ambientes--999"
        )
        # fecha_scrape es ISO con timezone
        fecha = datetime.fromisoformat(resultado["fecha_scrape"])
        assert fecha.tzinfo is not None

    def test_tarjeta_minima_con_swap_capital_federal(self) -> None:
        listing = _listing_del_html(HTML_TARJETA_MINIMA)
        resultado = parsear_listing(listing)

        assert resultado is not None
        assert resultado["id"] == "1000"
        assert resultado["tipo_propiedad"] == "casa"
        assert resultado["precio"] == "150000"
        # Fallback de moneda por idmoneda (1 = ARS)
        assert resultado["moneda"] == "ARS"
        # "Capital Federal" como último segmento: se swapea barrio/sub_barrio
        assert resultado["barrio"] == "Palermo"
        assert resultado["sub_barrio"] is None

    def test_sin_id_devuelve_none(self) -> None:
        html = """
        <div class="listing__item">
          <a montooperacion="100000" href="/departamento-en-venta-en-x--1">
            <div class="card__title--primary">Departamento en Venta en X, Y</div>
          </a>
        </div>
        """
        listing = _listing_del_html(html)
        assert parsear_listing(listing) is None

    def test_sin_link_devuelve_none(self) -> None:
        html = '<div class="listing__item"><span>sin link</span></div>'
        listing = _listing_del_html(html)
        assert parsear_listing(listing) is None

    def test_ambientes_de_la_tarjeta_cuando_url_no_tiene(self) -> None:
        html = """
        <div class="listing__item">
          <a data-item-card="77" montooperacion="100000"
             href="/ph-en-venta-en-villa-crespo--77">
            <div class="card__title--primary">PH en Venta en Villa Crespo</div>
            <ul class="card__main-features">
              <li><i class="basico1-icon-cantidad_ambientes"></i><span>2</span></li>
            </ul>
          </a>
        </div>
        """
        listing = _listing_del_html(html)
        resultado = parsear_listing(listing)
        assert resultado is not None
        assert resultado["ambientes"] == "2"
        assert resultado["tipo_propiedad"] == "ph"


class TestHelpersCsv:
    def test_cargar_ids_existentes_con_archivo_vacio(self, tmp_path: Path) -> None:
        ruta = tmp_path / "listados.csv"
        assert cargar_ids_existentes(str(ruta)) == set()

    def test_ciclo_completo_guardar_y_cargar(self, tmp_path: Path) -> None:
        ruta = tmp_path / "listados.csv"

        asegurar_encabezado(str(ruta))
        filas = [
            {"id": "1", "titulo": "A", "precio": "100"},
            {"id": "2", "titulo": "B", "precio": "200"},
        ]
        guardar_filas(str(ruta), filas)

        assert cargar_ids_existentes(str(ruta)) == {"1", "2"}

        with open(ruta, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            filas_leidas = list(reader)

        assert reader.fieldnames == COLUMNS
        assert len(filas_leidas) == 2
        assert filas_leidas[0]["id"] == "1"

    def test_guardar_filas_apenda(self, tmp_path: Path) -> None:
        ruta = tmp_path / "listados.csv"

        asegurar_encabezado(str(ruta))
        guardar_filas(str(ruta), [{"id": "1", "titulo": "A"}])
        guardar_filas(str(ruta), [{"id": "2", "titulo": "B"}])

        assert cargar_ids_existentes(str(ruta)) == {"1", "2"}
