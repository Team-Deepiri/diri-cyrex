"""API key authorization policy for the Cyrex HTTP surface.

Deliberately free of framework and settings imports so the policy can be unit
tested without standing up the app or its dependencies.
"""

from __future__ import annotations

import hmac
from typing import Optional

# Served before authentication so orchestrators and probes can reach a server
# whose API key is missing or misconfigured.
UNAUTHENTICATED_PATH_PREFIXES = ("/health", "/metrics")

# Shipped in .env.example. Treated as "no key configured" so a copied example
# file cannot leave a deployment open.
PLACEHOLDER_API_KEY = "change-me"

UNCONFIGURED_DETAIL = (
    "Server authentication is not configured. Set CYREX_API_KEY to a generated "
    "secret, or set CYREX_ALLOW_INSECURE_AUTH=true for local development only."
)

AuthDenial = tuple[int, str]


def api_key_configured(configured_key: Optional[str]) -> bool:
    """True when a real secret is set, as opposed to unset or the placeholder."""
    return bool(configured_key) and configured_key != PLACEHOLDER_API_KEY


def evaluate_request(
    *,
    method: str,
    path: str,
    provided_key: Optional[str],
    configured_key: Optional[str],
    allow_insecure: bool = False,
) -> Optional[AuthDenial]:
    """Return ``(status_code, detail)`` when a request must be rejected.

    Every route outside ``UNAUTHENTICATED_PATH_PREFIXES`` requires a valid
    ``x-api-key``. There is no client-asserted bypass: a header cannot vouch for
    itself, so honoring one is equivalent to having no authentication at all.

    An unconfigured key fails closed with 503 rather than serving traffic, since
    the alternative silently turns a fresh or misconfigured deployment into an
    open one.
    """
    if method == "OPTIONS":
        return None
    if path.startswith(UNAUTHENTICATED_PATH_PREFIXES):
        return None

    if not api_key_configured(configured_key):
        if allow_insecure:
            return None
        return (503, UNCONFIGURED_DETAIL)

    if not hmac.compare_digest(provided_key or "", configured_key or ""):
        return (401, "Invalid API key")

    return None
