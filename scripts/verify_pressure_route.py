"""Verify the live database-backed pressure route with temporary data.

Usage from the repository root::

    python scripts/verify_pressure_route.py
    python scripts/verify_pressure_route.py --start-server

By default the API must already be running at ``--api-url``.  The script
seeds through ``PressureEngine``, reads through the HTTP route, validates the
response, and removes all rows for its temporary document in a ``finally``
block.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from app.database.postgres import PostgreSQLManager  # noqa: E402
from app.pipeline.contracts.pressure_events import (  # noqa: E402
    DuelDisagreement,
    LowConfidenceField,
    PassDiscrepancy,
)
from app.pipeline.pressure.engine import PressureEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-host", default=os.getenv("POSTGRES_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5434")))
    parser.add_argument("--db-name", default=os.getenv("POSTGRES_DB", "cyrex_db"))
    parser.add_argument("--db-user", default=os.getenv("POSTGRES_USER", "deepiri_cyrex"))
    parser.add_argument(
        "--db-password",
        default=os.getenv("POSTGRES_PASSWORD", "deepiripassword"),
    )
    parser.add_argument("--start-server", action="store_true")
    return parser.parse_args()


async def wait_for_api(
    client: httpx.AsyncClient,
    api_url: str,
    server: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            error = server.stderr.read() if server.stderr else ""
            raise RuntimeError(f"API process exited before startup:\n{error}")
        try:
            response = await client.get(f"{api_url}/api/v1/pressure/__route_check_ready__")
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"API did not become reachable at {api_url}")


async def main() -> None:
    args = parse_args()
    document_id = f"pressure_route_check_{uuid4().hex}"
    server: subprocess.Popen[str] | None = None
    db = PostgreSQLManager(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
        min_size=1,
        max_size=2,
    )

    try:
        if args.start_server:
            environment = os.environ.copy()
            environment.update(
                {
                    "POSTGRES_HOST": args.db_host,
                    "POSTGRES_PORT": str(args.db_port),
                    "POSTGRES_DB": args.db_name,
                    "POSTGRES_USER": args.db_user,
                    "POSTGRES_PASSWORD": args.db_password,
                }
            )
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

        initialized = await db.initialize(max_retries=3, retry_delay=0.5)
        if not initialized:
            raise RuntimeError(
                f"Could not connect to PostgreSQL at {args.db_host}:{args.db_port}/{args.db_name}"
            )
        await PressureEngine(db).accept_many(
            [
                PassDiscrepancy(
                    document_id=document_id,
                    section_id="financial_terms",
                    page=1,
                    artifact_id=f"{document_id}_rent",
                    field_name="base_rent",
                    pass_a_value=4500,
                    pass_b_value=4600,
                ),
                DuelDisagreement(
                    document_id=document_id,
                    section_id="financial_terms",
                    page=1,
                    artifact_id=f"{document_id}_duel",
                    field_name="notice_period",
                ),
                LowConfidenceField(
                    document_id=document_id,
                    section_id="financial_terms",
                    page=1,
                    artifact_id=f"{document_id}_confidence",
                    field_name="maintenance_obligation",
                    confidence=0.52,
                ),
            ]
        )

        async with httpx.AsyncClient(timeout=15) as client:
            if args.start_server:
                await wait_for_api(client, args.api_url, server)
            response = await client.get(f"{args.api_url}/api/v1/pressure/{document_id}")
            response.raise_for_status()
            payload = response.json()

        cells = payload["cells"]
        assert payload["document_id"] == document_id
        assert payload["fault_zone_count"] == 1
        assert payload["max_score"] == 0.75
        assert len(cells) == 1
        assert cells[0]["is_fault_zone"] is True
        assert cells[0]["low_confidence_count"] == 1
        assert len(cells[0]["drill_down_artifact_ids"]) == 3
        print(f"Pressure route verification passed for {document_id}")
        print(payload)
    finally:
        if db._pool is not None:
            await db.execute(
                "DELETE FROM cyrex.pressure_cell_artifacts WHERE document_id = $1",
                document_id,
            )
            await db.execute(
                "DELETE FROM cyrex.pressure_cell_metrics WHERE document_id = $1",
                document_id,
            )
            await db.execute(
                "DELETE FROM cyrex.pressure_cells WHERE document_id = $1",
                document_id,
            )
            await db.execute(
                "DELETE FROM cyrex.pressure_events WHERE document_id = $1",
                document_id,
            )
            await db.close()
        if server is not None:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
