"""Corpus export — deepiri-dataset-processor presets before Helox emit."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

logger = get_logger("cyrex.pipeline.corpus_exporter")

try:
    from deepiri_dataset_processor.export.cyrex_bridge import (
        batch_to_jsonl_records,
        correction_to_training,
        reckoning_record_to_training,
        visual_observation_to_training,
    )
    from deepiri_dataset_processor.pipeline.presets.cyrex_agi import (
        cyrex_reckoning_export_preset,
        cyrex_visual_grounding_preset,
        export_training_jsonl,
    )
except ImportError:  # pragma: no cover
    batch_to_jsonl_records = None  # type: ignore
    correction_to_training = None  # type: ignore
    reckoning_record_to_training = None  # type: ignore
    visual_observation_to_training = None  # type: ignore
    cyrex_reckoning_export_preset = None  # type: ignore
    cyrex_visual_grounding_preset = None  # type: ignore
    export_training_jsonl = None  # type: ignore


class CorpusExporter:
    """Prepare AGI training rows through dataset-processor quality gates."""

    def __init__(self, *, export_dir: Optional[str] = None) -> None:
        self._export_dir = export_dir or os.environ.get(
            "CYREX_TRAINING_EXPORT_DIR",
            tempfile.gettempdir(),
        )

    def available(self) -> bool:
        return reckoning_record_to_training is not None

    def reckoning_rows(
        self,
        records: List[Any],
        *,
        document_id: str,
        artifact_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.available():
            return []
        rows: List[Dict[str, Any]] = []
        for rec in records:
            payload = rec.model_dump(mode="json") if hasattr(rec, "model_dump") else dict(rec)
            rows.append(
                reckoning_record_to_training(
                    payload,
                    document_id=document_id,
                    artifact_id=artifact_id,
                )
            )
        return self._finalize(rows, preset_kind="reckoning")

    def correction_row(
        self,
        *,
        document_id: str,
        field_name: str,
        corrected_value: Any,
        original_value: Any = None,
        actor_id: str = "unknown",
        artifact_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.available():
            return []
        row = correction_to_training(
            document_id=document_id,
            field_name=field_name,
            corrected_value=corrected_value,
            original_value=original_value,
            actor_id=actor_id,
            artifact_id=artifact_id,
        )
        return self._finalize([row], preset_kind="reckoning")

    def visual_rows(
        self,
        traces: List[Dict[str, Any]],
        *,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        if not self.available():
            return []
        rows = [
            visual_observation_to_training(t, document_id=document_id) for t in traces
        ]
        return self._finalize(rows, preset_kind="visual")

    def export_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        preset_kind: str = "visual",
    ) -> List[Dict[str, Any]]:
        """Run pre-built training rows through dataset-processor quality gates."""
        if not self.available() or not rows:
            return rows
        return self._finalize(rows, preset_kind=preset_kind)

    def _finalize(
        self, rows: List[Dict[str, Any]], *, preset_kind: str
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []
        normalized = batch_to_jsonl_records(rows)
        preset = (
            cyrex_visual_grounding_preset()
            if preset_kind == "visual"
            else cyrex_reckoning_export_preset()
        )
        try:
            doc_slug = document_id_from_rows(normalized)
            out_path = os.path.join(
                self._export_dir,
                f"cyrex-{preset_kind}-{doc_slug}.jsonl",
            )
            stats = export_training_jsonl(normalized, out_path, preset=preset)
            logger.info(
                "corpus export finalized",
                extra={"path": stats.get("path"), "count": stats.get("record_count")},
            )
            stage_result = preset.run(normalized)
            if stage_result.success and stage_result.processed_data:
                data = stage_result.processed_data.data
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.warning("corpus export preset failed, using raw rows: %s", exc)
        return normalized


def document_id_from_rows(rows: List[Dict[str, Any]]) -> str:
    for row in rows:
        meta = row.get("metadata") or {}
        doc = meta.get("document_id")
        if doc:
            return str(doc).replace("/", "_")[:64]
    return "batch"
