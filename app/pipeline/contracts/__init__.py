"""Shared contract layer for the Cyrex Artifact Engine.

This package defines the types, protocols, and event schemas
that all tracks (A–D) depend on. Track implementations import
from here — they must not import each other.
"""

from app.pipeline.contracts import models, ports, pressure_events

__all__ = ["models", "ports", "pressure_events"]
