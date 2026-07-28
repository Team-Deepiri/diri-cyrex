"""
Import-time checks for modules loaded before uvicorn can serve traffic.

These are intentionally narrow integration smokes (not unit tests): they catch
missing symbols, stale deepiri-modelkit pins, langchain-milvus wiring, and
protobuf/gRPC stub mismatches that `ruff` on pipeline-only paths will miss.
"""


def test_modelkit_sidecar_utils_available():
    from deepiri_modelkit.streaming.sidecar_utils import (
        env_float,
        resolve_grpc_addr,
        sidecar_payload_from_fields,
    )

    assert env_float("__MISSING__", 2.5) == 2.5
    assert resolve_grpc_addr("http://synapse-sidecar:8081") == "synapse-sidecar:50051"
    assert sidecar_payload_from_fields({"event_type": "ping", "payload": "{}"})["event"] == "ping"


def test_langchain_vectorstore_imports():
    from langchain_community.vectorstores import Chroma
    from langchain_milvus import Milvus

    assert Chroma is not None
    assert Milvus is not None


def test_sugar_glider_protobuf_stubs_compatible_with_runtime():
    """Loads checked-in gRPC stubs (protobuf 7.x gencode) without integrations package init."""
    import sys
    from pathlib import Path

    gen_root = (
        Path(__file__).resolve().parents[2] / "app" / "integrations" / "streaming" / "gen"
    )
    sys.path.insert(0, str(gen_root))
    from proto.synapse.v1 import sugar_glider_pb2  # noqa: F401
    from proto.synapse.v1 import sugar_glider_pb2_grpc  # noqa: F401


def test_artifact_routes_importable():
    from app.routes.artifacts import router

    assert router.prefix == "/api/v1/artifacts"


def test_fastapi_app_importable():
    from app.main import app

    assert app.title == "Deepiri AI Challenge Service API"
