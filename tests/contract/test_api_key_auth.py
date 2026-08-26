"""Tests for the API key authorization policy.

Regression coverage for the ``x-desktop-client`` bypass: sending that header
with no key previously satisfied the guard on every route.
"""

from __future__ import annotations

import pytest

from app.api_key_auth import (
    PLACEHOLDER_API_KEY,
    api_key_configured,
    evaluate_request,
)

REAL_KEY = "s3cret-generated-key"


def _evaluate(
    *,
    method: str = "GET",
    path: str = "/api/v1/agent/chat",
    provided_key=None,
    configured_key: str = REAL_KEY,
    allow_insecure: bool = False,
):
    return evaluate_request(
        method=method,
        path=path,
        provided_key=provided_key,
        configured_key=configured_key,
        allow_insecure=allow_insecure,
    )


class TestDesktopClientBypass:
    def test_no_header_grants_access_without_a_key(self):
        """A client-asserted header must never stand in for a credential."""
        denial = _evaluate(provided_key=None)
        assert denial is not None
        assert denial[0] == 401

    def test_wrong_key_is_rejected(self):
        denial = _evaluate(provided_key="not-the-key")
        assert denial is not None
        assert denial[0] == 401

    def test_valid_key_is_accepted(self):
        assert _evaluate(provided_key=REAL_KEY) is None

    def test_empty_key_is_rejected(self):
        denial = _evaluate(provided_key="")
        assert denial is not None
        assert denial[0] == 401


class TestUnauthenticatedPaths:
    @pytest.mark.parametrize(
        "path",
        ["/health", "/health/detailed", "/metrics"],
    )
    def test_probe_paths_stay_open(self, path):
        assert _evaluate(path=path, provided_key=None) is None

    def test_probe_paths_open_even_when_unconfigured(self):
        assert (
            _evaluate(path="/health", provided_key=None, configured_key=None) is None
        )

    def test_options_preflight_is_allowed(self):
        assert _evaluate(method="OPTIONS", provided_key=None) is None

    def test_lookalike_path_is_not_treated_as_public(self):
        denial = _evaluate(path="/api/v1/health-report", provided_key=None)
        assert denial is not None
        assert denial[0] == 401


class TestUnconfiguredKeyFailsClosed:
    @pytest.mark.parametrize("configured_key", [None, "", PLACEHOLDER_API_KEY])
    def test_missing_or_placeholder_key_returns_503(self, configured_key):
        denial = _evaluate(provided_key=None, configured_key=configured_key)
        assert denial is not None
        assert denial[0] == 503

    def test_placeholder_key_is_not_usable_as_a_credential(self):
        denial = _evaluate(
            provided_key=PLACEHOLDER_API_KEY, configured_key=PLACEHOLDER_API_KEY
        )
        assert denial is not None
        assert denial[0] == 503

    def test_insecure_escape_hatch_allows_local_development(self):
        assert (
            _evaluate(provided_key=None, configured_key=None, allow_insecure=True)
            is None
        )

    def test_escape_hatch_does_not_weaken_a_configured_key(self):
        denial = _evaluate(
            provided_key="not-the-key",
            configured_key=REAL_KEY,
            allow_insecure=True,
        )
        assert denial is not None
        assert denial[0] == 401


class TestApiKeyConfigured:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, False),
            ("", False),
            (PLACEHOLDER_API_KEY, False),
            (REAL_KEY, True),
        ],
    )
    def test_configured_detection(self, value, expected):
        assert api_key_configured(value) is expected
