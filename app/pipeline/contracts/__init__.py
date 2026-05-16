"""Shared contract layer for the Cyrex Artifact Engine.

This package defines the types, protocols, and event schemas
that all tracks (A–D) depend on. Track implementations import
from here — they must not import each other.

Frozen JSON Schema for REST/MCP payloads lives under ``json_schema/``;
regenerate via ``scripts/export_cyrex_contract_json_schema.py`` after model changes.
"""

from app.pipeline.contracts import models, ports, pressure_events

__all__ = ["models", "ports", "pressure_events"]
