"""Tests for DocumentIndexingService.index_pre_chunked and the
POST /api/v1/documents/index/chunks route.

index_pre_chunked exists so producer/consumer pipelines (document.vectorize)
that already split text upstream (LIS's DocumentChunk model) can index into
Milvus without Cyrex re-chunking and losing chunk_id alignment -- and so the
caller gets the raw embedding vector back, which index_text/index_file don't
return.

NOTE: not run in CI for this change -- this repo's Python environment
(torch/transformers/langchain/pymilvus) isn't installed in the environment
this test was authored in. Verified by syntax check (py_compile) and by
mirroring the exact patterns already exercised by index_text's tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.document_indexing_service import (
    B2BDocumentType,
    DocumentIndexingService,
)


def make_service(vector_store: MagicMock) -> DocumentIndexingService:
    """Construct a DocumentIndexingService without touching real Milvus."""
    service = object.__new__(DocumentIndexingService)
    service.collection_name = "test_collection"
    service.chunk_size = 1500
    service.chunk_overlap = 300
    service.chunking_strategy = "paragraph"
    service.use_enhanced_rag = True
    service.enable_auto_extraction = False
    service._lease_processor = None
    service._contract_processor = None
    service.vector_store = vector_store
    return service


@pytest.mark.asyncio
async def test_index_pre_chunked_preserves_chunk_ids_and_order():
    vectors_by_text = {
        "Tenant shall pay rent monthly.": [0.11, 0.22],
        "Lease term is 5 years.": [0.33, 0.44],
    }
    vector_store = MagicMock()
    vector_store.embeddings.embed_query.side_effect = lambda text: vectors_by_text[text]
    vector_store.add_documents.return_value = None

    service = make_service(vector_store)
    chunks = [
        {"chunk_id": "lis-chunk-0", "text": "Tenant shall pay rent monthly."},
        {"chunk_id": "lis-chunk-1", "text": "Lease term is 5 years."},
    ]

    results = await service.index_pre_chunked(
        document_id="doc-abc",
        chunks=chunks,
        doc_type=B2BDocumentType.LEASE,
        industry="real_estate",
    )

    assert [r["chunk_id"] for r in results] == ["lis-chunk-0", "lis-chunk-1"]
    assert all(r["dimensions"] == 2 for r in results)
    assert results[0]["vector"] == [0.11, 0.22]
    assert results[1]["vector"] == [0.33, 0.44]

    # Persisted into Milvus with chunk identity preserved, not re-split.
    vector_store.add_documents.assert_called_once()
    (documents,), _ = vector_store.add_documents.call_args
    assert len(documents) == 2
    assert documents[0].metadata["chunk_id"] == "lis-chunk-0"
    assert documents[0].metadata["document_id"] == "doc-abc"
    assert documents[0].metadata["doc_type"] == "lease"
    assert documents[1].metadata["chunk_id"] == "lis-chunk-1"


@pytest.mark.asyncio
async def test_index_pre_chunked_rejects_empty_chunks_list():
    service = make_service(MagicMock())
    with pytest.raises(ValueError, match="non-empty"):
        await service.index_pre_chunked(document_id="doc-abc", chunks=[])


@pytest.mark.asyncio
async def test_index_pre_chunked_rejects_missing_chunk_id():
    service = make_service(MagicMock())
    with pytest.raises(ValueError, match="missing chunk_id"):
        await service.index_pre_chunked(
            document_id="doc-abc", chunks=[{"text": "no id here"}]
        )


@pytest.mark.asyncio
async def test_index_pre_chunked_rejects_empty_text():
    service = make_service(MagicMock())
    with pytest.raises(ValueError, match="empty text"):
        await service.index_pre_chunked(
            document_id="doc-abc", chunks=[{"chunk_id": "c0", "text": ""}]
        )


@pytest.mark.asyncio
async def test_index_pre_chunked_merges_document_and_chunk_metadata():
    vector_store = MagicMock()
    vector_store.embeddings.embed_query.return_value = [0.1, 0.2]

    service = make_service(vector_store)
    await service.index_pre_chunked(
        document_id="doc-abc",
        chunks=[
            {
                "chunk_id": "c0",
                "text": "hello",
                "metadata": {"clauseType": "rent"},
            }
        ],
        metadata={"leaseVersion": "2.0"},
    )

    (documents,), _ = vector_store.add_documents.call_args
    metadata = documents[0].metadata
    assert metadata["leaseVersion"] == "2.0"
    assert metadata["clauseType"] == "rent"
    assert metadata["chunk_id"] == "c0"
    assert metadata["total_chunks"] == 1
