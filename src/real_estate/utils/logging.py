"""
Configuración centralizada de logging.

Reemplaza los ``print()`` dispersos por ``logging`` estructurado con un
formato consistente.  Cada módulo crea su propio logger con
``logging.getLogger(__name__)`` y la función ``configurar_logging`` se llama
una sola vez al inicio de cada entry point (scripts).

Uso típico en un script:

    from real_estate.utils.logging import configurar_logging
    configurar_logging()
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

#: Nombre del logger raíz del proyecto.
LOGGER_NAME = "real_estate"

#: Formato por defecto de los mensajes de log.
FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Formato de la fecha en los mensajes de log.
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


def configurar_logging(
    nivel: int = logging.INFO,
    formato: str = FORMATO,
    formato_fecha: str = FORMATO_FECHA,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configura el logging para el proyecto ``real_estate``.

    Crea un handler en ``stream`` (por defecto ``sys.stderr``) con el
    ``formato`` dado y lo adjunta al logger raíz ``real_estate``.  Los
    handlers y el nivel se configuren de modo que los mensajes de
    ``real_estate.*`` se muestren, mientras que los de terceros (p. ej.
    ``urllib3``, ``matplotlib``) se silencian o suben de nivel.

    Parámetros
    ----------
    nivel:
        Nivel mínimo de los mensajes que se muestran (default ``INFO``).
    formato:
        Formato de los mensajes (ver ``FORMATO``).
    formato_fecha:
        Formato de la marca de tiempo.
    stream:
        Stream de salida; por defecto ``sys.stderr``.

    Devuelve
    --------
    logging.Logger
        El logger raíz de ``real_estate`` ya configurado.
    """

    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(nivel)

    formatter = logging.Formatter(formato, datefmt=formato_fecha)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Silenciar loggers de terceros que son muy verbosos.
    for nombre_ruidoso in ("urllib3", "matplotlib", "matplotlib.font_manager"):
        logging.getLogger(nombre_ruidoso).setLevel(logging.WARNING)

    return logger
