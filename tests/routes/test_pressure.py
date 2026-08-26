from __future__ import annotations

import pytest

from app.pipeline.contracts.models import PressureCell
from app.routes.pressure import get_document_pressure


class PressureStore:
    async def get_pressure(self, document_id=None):
        return [
            PressureCell(
                document_id=document_id or "doc-1",
                section_id="section-1",
                score=0.9,
                is_fault_zone=True,
            ),
            PressureCell(
                document_id=document_id or "doc-1",
                section_id="section-2",
                score=0.3,
                is_fault_zone=False,
            ),
        ]


@pytest.mark.asyncio
async def test_pressure_route_returns_stable_map_response():
    response = await get_document_pressure("doc-1", PressureStore())

    assert response.model_dump() == {
        "document_id": "doc-1",
        "cells": [
            {
                "document_id": "doc-1",
                "section_id": "section-1",
                "page": None,
                "discrepancy_count": 0,
                "reflect_failures": 0,
                "low_confidence_count": 0,
                "duel_disagreements": 0,
                "score": 0.9,
                "is_fault_zone": True,
                "drill_down_artifact_ids": [],
            },
            {
                "document_id": "doc-1",
                "section_id": "section-2",
                "page": None,
                "discrepancy_count": 0,
                "reflect_failures": 0,
                "low_confidence_count": 0,
                "duel_disagreements": 0,
                "score": 0.3,
                "is_fault_zone": False,
                "drill_down_artifact_ids": [],
            },
        ],
        "fault_zone_count": 1,
        "max_score": 0.9,
    }
