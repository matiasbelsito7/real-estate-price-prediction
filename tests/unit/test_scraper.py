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

import pytest
from bs4 import BeautifulSoup

from real_estate.ingestion.scraper import (
    BASE_URL,
    COLUMNS,
    MAX_PAGINAS_SERVICIO,
    STATUS_BLOQUEO,
    asegurar_encabezado,
    cargar_ids_existentes,
    cargar_progreso,
    construir_url_pagina,
    construir_url_segmento,
    detectar_tipo_propiedad,
    extraer_ambientes_de_url,
    extraer_features_de_tarjeta,
    guardar_filas,
    guardar_progreso,
    pagina_de_reanudacion,
    parsear_listing,
    scrapear,
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


class TestConstruirUrlSegmento:
    def test_sin_filtros_usa_la_base(self) -> None:
        assert construir_url_segmento() == BASE_URL

    def test_con_tipo(self) -> None:
        url = construir_url_segmento(tipo="departamentos")
        assert url == "https://www.argenprop.com/inmuebles/departamentos-en-venta/capital-federal"

    def test_con_barrio(self) -> None:
        assert construir_url_segmento(barrio="palermo") == f"{BASE_URL}/palermo"

    def test_tipo_y_barrio(self) -> None:
        url = construir_url_segmento(tipo="casas", barrio="palermo")
        assert url == "https://www.argenprop.com/inmuebles/casas-en-venta/capital-federal/palermo"


class TestConstruirUrlPaginaConSegmento:
    def test_pagina_uno_es_la_base_del_segmento(self) -> None:
        assert construir_url_pagina(1, base_url="https://x/segmento") == "https://x/segmento"

    def test_pagina_mayor_agrega_query_al_segmento(self) -> None:
        assert (
            construir_url_pagina(3, base_url="https://x/segmento") == "https://x/segmento?pagina-3"
        )


class TestProgreso:
    def test_guardar_y_cargar(self, tmp_path: Path) -> None:
        ruta = tmp_path / "progreso.json"
        estado = {"palermo": {"pagina": 42, "completo": False}}
        guardar_progreso(str(ruta), estado)
        assert cargar_progreso(str(ruta)) == estado

    def test_cargar_sin_archivo(self) -> None:
        assert cargar_progreso("no-existe.json") == {}

    def test_cargar_corrupto(self, tmp_path: Path) -> None:
        ruta = tmp_path / "progreso.json"
        ruta.write_text("{json roto", encoding="utf-8")
        assert cargar_progreso(str(ruta)) == {}

    def test_guardar_crea_el_directorio(self, tmp_path: Path) -> None:
        ruta = tmp_path / "anidado" / "progreso.json"
        guardar_progreso(str(ruta), {"x": {"pagina": 1, "completo": True}})
        assert cargar_progreso(str(ruta)) == {"x": {"pagina": 1, "completo": True}}

    def test_pagina_de_reanudacion_sin_progreso(self) -> None:
        assert pagina_de_reanudacion("no-existe.json", "palermo") is None

    def test_pagina_de_reanudacion_completo(self, tmp_path: Path) -> None:
        ruta = tmp_path / "progreso.json"
        guardar_progreso(str(ruta), {"palermo": {"pagina": 99, "completo": True}})
        assert pagina_de_reanudacion(str(ruta), "palermo") is None

    def test_pagina_de_reanudacion_parcial(self, tmp_path: Path) -> None:
        ruta = tmp_path / "progreso.json"
        guardar_progreso(str(ruta), {"palermo": {"pagina": 42, "completo": False}})
        assert pagina_de_reanudacion(str(ruta), "palermo") == 43


# ---------------------------------------------------------------------------
# Helpers para simular el sitio en los tests de scrapear()
# ---------------------------------------------------------------------------


class FakeRespuesta:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeSesion:
    """Sesión fake: entrega las respuestas en cola y, si se agota, repite la
    última (como un servidor que sigue bloqueando)."""

    def __init__(self, respuestas: list[FakeRespuesta]) -> None:
        self.respuestas = list(respuestas)
        self.ultima = respuestas[-1] if respuestas else FakeRespuesta(404)
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, url: str, timeout: int | None = None) -> FakeRespuesta:
        self.calls += 1
        if self.respuestas:
            self.ultima = self.respuestas.pop(0)
        return self.ultima


def html_de_una_pagina(n_avisos: int, id_inicial: int = 1000) -> str:
    """HTML de listado con `n_avisos` tarjetas, ids únicos consecutivos."""
    tarjeta = """
    <div class="listing__item">
      <a data-item-card="{id}" montooperacion="250000" montonormalizado="250000"
         idmoneda="2" dormitorios="2" idtipopropiedad="1"
         href="/departamento-en-venta-en-palermo-3-ambientes--{id}">
         <div class="card__title--primary">Departamento en Venta en Palermo Chico, Palermo</div>
         <div class="card__currency">USD</div>
         <ul class="card__main-features">
           <li><i class="basico1-icon-cantidad_dormitorios"></i><span>2</span></li>
           <li><i class="basico1-icon-superficie_cubierta"></i><span>90 m2 cubie.</span></li>
         </ul>
      </a>
    </div>
    """
    return "\n".join(tarjeta.format(id=id_inicial + i) for i in range(n_avisos))


def configurar_fake(monkeypatch: pytest.MonkeyPatch, sesion: FakeSesion) -> None:
    """Parchea Session, sleeps y delays para que scrapear() corra sin red."""
    # Forma con string: parchea el módulo global, que es el que usa el scraper.
    monkeypatch.setattr("requests.Session", lambda: sesion)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr("random.uniform", lambda _a, _b: 0.0)


class TestScrapearCon202:
    def test_202_en_pagina_100_corta_el_segmento_como_completo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        respuestas = [FakeRespuesta(200, html_de_una_pagina(1, 1000 + i)) for i in range(99)]
        respuestas.append(FakeRespuesta(STATUS_BLOQUEO, ""))
        sesion = FakeSesion(respuestas)
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=None,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            base_url=construir_url_segmento(barrio="palermo"),
            nombre_segmento="palermo",
            archivo_progreso=str(prog_path),
        )

        # 99 páginas OK (pagina 100 devuelve 202 = cap del sitio)
        assert len(cargar_ids_existentes(str(csv_path))) == 99
        assert sesion.calls == 100
        estado = cargar_progreso(str(prog_path))
        assert estado["palermo"]["completo"] is True
        assert estado["palermo"]["pagina"] == 99

    def test_202_en_pagina_temprana_reintenta_y_avanza(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        # pagina 1 OK, pagina 2 primero 202 y después 200, paginas 3 y 4 OK
        respuestas = [
            FakeRespuesta(200, html_de_una_pagina(1, 1000)),
            FakeRespuesta(STATUS_BLOQUEO, ""),
            FakeRespuesta(200, html_de_una_pagina(1, 1001)),
            FakeRespuesta(200, html_de_una_pagina(1, 1002)),
            FakeRespuesta(200, html_de_una_pagina(1, 1003)),
        ]
        sesion = FakeSesion(respuestas)
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=4,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            nombre_segmento="palermo",
            archivo_progreso=str(prog_path),
        )

        # 4 páginas procesadas; el 202 en la página 2 requirió un request extra
        assert len(cargar_ids_existentes(str(csv_path))) == 4
        assert sesion.calls == 5
        assert cargar_progreso(str(prog_path))["palermo"]["completo"] is True

    def test_bloqueo_sostenido_deja_el_segmento_incompleto(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        # pagina 1 OK; el resto queda bloqueado (202) de forma persistente
        respuestas = [FakeRespuesta(200, html_de_una_pagina(1, 1000))]
        respuestas.append(FakeRespuesta(STATUS_BLOQUEO, ""))
        sesion = FakeSesion(respuestas)
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=None,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            nombre_segmento="palermo",
            archivo_progreso=str(prog_path),
            reintentos_202=2,
        )

        # Se guardó la página 1 pero el segmento quedó incompleto para reanudar
        assert len(cargar_ids_existentes(str(csv_path))) == 1
        estado = cargar_progreso(str(prog_path))["palermo"]
        assert estado["completo"] is False
        assert estado["pagina"] == 1
        # Reanudar retomaría desde la página 2
        assert pagina_de_reanudacion(str(prog_path), "palermo") == 2

    def test_segmento_completo_en_progreso_se_salteca(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        guardar_progreso(str(prog_path), {"palermo": {"pagina": 99, "completo": True}})
        sesion = FakeSesion([FakeRespuesta(200, html_de_una_pagina(1, 1000))])
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=None,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            nombre_segmento="palermo",
            archivo_progreso=str(prog_path),
        )

        assert sesion.calls == 0
        assert cargar_ids_existentes(str(csv_path)) == set()

    def test_reanuda_desde_la_ultima_pagina_guardada(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        guardar_progreso(str(prog_path), {"palermo": {"pagina": 1, "completo": False}})
        # página 2 OK (id 2000), página 3 bloqueada para cortar la corrida
        sesion = FakeSesion(
            [
                FakeRespuesta(200, html_de_una_pagina(1, 2000)),
                FakeRespuesta(STATUS_BLOQUEO, ""),
            ]
        )
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=None,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            nombre_segmento="palermo",
            archivo_progreso=str(prog_path),
            reintentos_202=1,
        )

        # Reanudó en la página 2 (no volvió a pedir la 1)
        assert len(cargar_ids_existentes(str(csv_path))) == 1
        assert sesion.calls >= 2
        estado = cargar_progreso(str(prog_path))["palermo"]
        assert estado["completo"] is False
        assert estado["pagina"] == 2

    def test_cap_202_respetado_sin_importar_reintentos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Aunque el 202 en la página 100 es el cap del servidor, no se reintenta."""
        csv_path = tmp_path / "datos.csv"
        prog_path = tmp_path / "progreso.json"
        respuestas = [FakeRespuesta(200, html_de_una_pagina(1, 1000 + i)) for i in range(99)]
        respuestas.append(FakeRespuesta(STATUS_BLOQUEO, ""))
        sesion = FakeSesion(respuestas)
        configurar_fake(monkeypatch, sesion)

        scrapear(
            str(csv_path),
            max_paginas=None,
            pagina_inicio=1,
            delay_min=0.0,
            delay_max=0.0,
            nombre_segmento="global",
            archivo_progreso=str(prog_path),
            reintentos_202=10,
        )

        # Un solo request a la página 100: el cap se detecta sin reintentar
        assert sesion.calls == MAX_PAGINAS_SERVICIO
        assert cargar_progreso(str(prog_path))["global"]["completo"] is True
