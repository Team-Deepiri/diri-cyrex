#!/usr/bin/env python3
"""Emit frozen JSON Schema files from `app.pipeline.contracts` Pydantic models.

Run from repo root (diri-cyrex) with PYTHONPATH=. :

    PYTHONPATH=. python scripts/export_cyrex_contract_json_schema.py

Regenerate after changing contract models so REST/MCP surfaces stay aligned.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "pipeline" / "contracts" / "json_schema"


def main() -> None:
    # Import after ROOT is known (app package lives under ROOT).
    from app.pipeline.contracts.models import (
        ArtifactBundle,
        DuelState,
        PersonaScope,
        PredictionRecord,
        PressureCell,
        ReflectionResult,
        SynthesisResult,
    )
    from app.pipeline.contracts.pressure_events import PressureEvent

    OUT.mkdir(parents=True, exist_ok=True)

    exports: list[tuple[str, type]] = [
        ("artifact_bundle", ArtifactBundle),
        ("duel_state", DuelState),
        ("prediction_record", PredictionRecord),
        ("pressure_cell", PressureCell),
        ("pressure_event", PressureEvent),
        ("persona_scope", PersonaScope),
        ("reflection_result", ReflectionResult),
        ("synthesis_result", SynthesisResult),
    ]

    for name, model in exports:
        schema = TypeAdapter(model).json_schema()
        path = OUT / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
