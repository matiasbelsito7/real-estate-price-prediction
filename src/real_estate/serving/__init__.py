"""Servicio de predicción del modelo entrenado (Fase 10)."""

from __future__ import annotations

from real_estate.serving.modelo import ModeloPrediccion
from real_estate.serving.persistencia import cargar_bundle, guardar_bundle

__all__ = ["ModeloPrediccion", "cargar_bundle", "guardar_bundle"]
