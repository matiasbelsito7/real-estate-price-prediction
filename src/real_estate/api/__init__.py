"""API de predicción de precios de propiedades (Fase 10)."""

from __future__ import annotations

from real_estate.api.app import app, crear_app
from real_estate.api.config import ConfiguracionServicio
from real_estate.api.schemas import PrediccionSalida, PropiedadEntrada

__all__ = [
    "app",
    "crear_app",
    "ConfiguracionServicio",
    "PropiedadEntrada",
    "PrediccionSalida",
]
