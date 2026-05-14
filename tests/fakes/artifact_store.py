"""In-memory implementation of ArtifactStorePort for track-local tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.pipeline.contracts.models import ArtifactBundle
from app.pipeline.contracts.ports import ArtifactStorePort


class InMemoryArtifactStore(ArtifactStorePort):
    """Simple in-memory store for testing. Thread-unsafe by design —
    each test creates its own instance.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ArtifactBundle] = {}
        self._by_doc: Dict[str, List[str]] = {}  # document_id -> [artifact_ids]

    async def create(self, bundle: ArtifactBundle) -> ArtifactBundle:
        self._store[bundle.artifact_id] = bundle
        self._by_doc.setdefault(bundle.document_id, []).append(bundle.artifact_id)
        return bundle

    async def get(self, artifact_id: str) -> Optional[ArtifactBundle]:
        bundle = self._store.get(artifact_id)
        if bundle and bundle.is_deleted:
            return None
        return bundle

    async def get_latest(
        self,
        document_id: str,
        artifact_type: Optional[str] = None,
    ) -> Optional[ArtifactBundle]:
        candidates = [
            self._store[aid]
            for aid in self._by_doc.get(document_id, [])
            if not self._store[aid].is_deleted
            and (artifact_type is None or self._store[aid].artifact_type.value == artifact_type)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.version)

    async def list_by_document(self, document_id: str) -> List[ArtifactBundle]:
        return [
            self._store[aid]
            for aid in self._by_doc.get(document_id, [])
            if not self._store[aid].is_deleted
        ]

    async def list_versions(self, document_id: str) -> List[int]:
        bundles = await self.list_by_document(document_id)
        return sorted({b.version for b in bundles})

    async def resolve_version(
        self,
        document_id: str,
        version: int,
    ) -> Optional[ArtifactBundle]:
        for aid in self._by_doc.get(document_id, []):
            b = self._store[aid]
            if not b.is_deleted and b.version == version:
                return b
        return None

    async def get_graph_neighborhood(
        self,
        artifact_id: str,
        hops: int = 1,
    ) -> Dict[str, Any]:
        """Return a simple neighborhood traversal (no real graph — just linked IDs)."""
        nodes: List[ArtifactBundle] = []
        edges: List[Dict[str, Any]] = []
        visited: set = set()

        async def _traverse(current: ArtifactBundle, depth: int) -> None:
            if depth > hops or current.artifact_id in visited:
                return
            visited.add(current.artifact_id)
            nodes.append(current)
            for ref_id in current.provenance.depends_on + current.provenance.depended_on_by:
                ref = self._store.get(ref_id)
                if ref and ref_id not in visited:
                    edges.append({
                        "from": current.artifact_id,
                        "to": ref_id,
                        "ref_type": "depends_on" if ref_id in current.provenance.depends_on
                                    else "depended_on_by",
                    })
                    await _traverse(ref, depth + 1)

        bundle = self._store.get(artifact_id)
        if bundle:
            await _traverse(bundle, 0)
        return {"nodes": nodes, "edges": edges}

    async def get_inverse_citations(
        self,
        document_id: str,
        char_start: int,
        char_end: int,
    ) -> List[ArtifactBundle]:
        """Find artifacts that cite the given span in the document."""
        results: List[ArtifactBundle] = []
        for bundle in self._by_doc.get(document_id, []):
            b = self._store.get(bundle)
            if b and any(
                c.locator.char_start == char_start and c.locator.char_end == char_end
                for c in b.citations
            ):
                results.append(b)
        return results
