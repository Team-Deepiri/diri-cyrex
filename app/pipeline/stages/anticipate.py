"""Dead reckoning anticipate stage — emits field priors before extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

from app.pipeline.contracts.models import PredictionRecord, PredictionStatus
from app.pipeline.contracts.ports import AnticipatePort

# Fields anticipate should predict per document class (v1 static templates).
# Tyler's parse stage may later drive this from extraction_templates.
DOCUMENT_CLASS_FIELDS: Dict[str, List[str]] = {
    "lease": [
        "base_rent",
        "security_deposit",
        "notice_period",
        "custom_addendum",
    ],
}


class PriorLookup(Protocol):
    """Corpus prior lookup — v1 in-memory; later backed by reckoning_field_priors."""

    def get_prior(self, document_class: str, field_name: str) -> Optional[PredictionRecord]:
        """Return a prior-only record for the field, or None if unknown."""


def prior_only(record: PredictionRecord) -> PredictionRecord:
    """Strip post-extraction fields; anticipate emits priors with status NO_PRIOR."""
    return PredictionRecord(
        field_name=record.field_name,
        predicted_range=record.predicted_range,
        predicted_mean=record.predicted_mean,
        actual_value=None,
        sigma_delta=None,
        status=PredictionStatus.NO_PRIOR,
        corpus_doc_count=record.corpus_doc_count,
        last_prior_update=record.last_prior_update,
    )


class InMemoryPriorLookup:
    """In-memory prior store keyed by (document_class, field_name)."""

    def __init__(self, priors: Dict[Tuple[str, str], PredictionRecord]) -> None:
        self._priors = priors

    def get_prior(self, document_class: str, field_name: str) -> Optional[PredictionRecord]:
        raw = self._priors.get((document_class, field_name))
        if raw is None:
            return None
        return prior_only(raw)


def default_lease_prior_lookup() -> InMemoryPriorLookup:
    """Seed priors aligned with tests/fixtures/cyrex_contracts/prediction_records.json."""
    last_update = datetime.fromisoformat("2024-06-10T08:00:00+00:00")
    priors: Dict[Tuple[str, str], PredictionRecord] = {
        ("lease", "base_rent"): PredictionRecord(
            field_name="base_rent",
            predicted_range={"min": 3800.0, "max": 5200.0},
            predicted_mean=4500.0,
            corpus_doc_count=147,
            last_prior_update=last_update,
        ),
        ("lease", "security_deposit"): PredictionRecord(
            field_name="security_deposit",
            predicted_range={"min": 4000.0, "max": 6000.0},
            predicted_mean=5000.0,
            corpus_doc_count=147,
            last_prior_update=last_update,
        ),
        ("lease", "notice_period"): PredictionRecord(
            field_name="notice_period",
            predicted_range={"min": 60.0, "max": 120.0},
            predicted_mean=90.0,
            corpus_doc_count=147,
            last_prior_update=last_update,
        ),
    }
    return InMemoryPriorLookup(priors)


class AnticipateStage(AnticipatePort):
    """Pre-extraction stage: lookup corpus priors per field for a document class.

    v1 ignores ``parsed_doc`` content until Tyler's parse stage lands a typed shape.
    Status tagging (confirmed/anomalous/novel) happens post-extract in reckoning pass.
    """

    def __init__(
        self,
        prior_lookup: Optional[PriorLookup] = None,
        field_templates: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._prior_lookup = prior_lookup or default_lease_prior_lookup()
        self._field_templates = field_templates or DOCUMENT_CLASS_FIELDS

    async def run(
        self,
        parsed_doc: Any,
        document_class: str,
    ) -> List[PredictionRecord]:
        del parsed_doc  # v1: priors driven by document_class + corpus, not parse output

        field_names = self._field_templates.get(document_class)
        if not field_names:
            return []

        records: List[PredictionRecord] = []
        for field_name in field_names:
            prior = self._prior_lookup.get_prior(document_class, field_name)
            if prior is not None:
                records.append(prior)
            else:
                records.append(
                    PredictionRecord(
                        field_name=field_name,
                        status=PredictionStatus.NO_PRIOR,
                    )
                )
        return records
