from __future__ import annotations

import subprocess
from typing import Any, Dict

import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient


def test_generate_bearer_token_success_sets_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_check_output(cmd, env=None, timeout=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = env or {}
        captured["timeout"] = timeout
        assert captured["env"].get("MCPGATEWAY_JWT_SECRET") == "abc123"
        return b"jwt-token"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    token = GatewayApiClient.generate_bearer_token(jwt_secret_key="abc123")
    assert token == "Bearer jwt-token"
    assert captured["timeout"] == 5.0


essential_env = {"MCPGATEWAY_JWT_SECRET": "k", "FOO": "bar"}


def test_generate_bearer_token_merges_extra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_output(cmd, env=None, timeout=None):  # type: ignore[no-untyped-def]
        for k, v in essential_env.items():
            assert env[k] == v
        return b"t"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    token = GatewayApiClient.generate_bearer_token(jwt_secret_key="k", extra_env={"FOO": "bar"})
    assert token == "Bearer t"


def test_init_auto_bearer_sets_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_output(cmd, env=None, timeout=None):  # type: ignore[no-untyped-def]
        return b"abc"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    client = GatewayApiClient("http://mock", auto_bearer=True, jwt_secret_key="s")
    assert client.auth_token == "Bearer abc"
    headers = client._headers()
    assert headers.get("Authorization") == "Bearer abc"


def test_init_auto_bearer_skipped_if_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.check_output should not be called")

    monkeypatch.setattr(subprocess, "check_output", raise_if_called)

    client = GatewayApiClient("http://mock", auth_token="Bearer pre", auto_bearer=True)
    assert client.auth_token == "Bearer pre"


def test_init_auto_bearer_skipped_if_token_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.check_output should not be called")

    monkeypatch.setattr(subprocess, "check_output", raise_if_called)

    client = GatewayApiClient("http://mock", token_provider=lambda: "Bearer dyn", auto_bearer=True)
    # _ensure_token should set the token via provider; no subprocess call
    client._ensure_token()
    assert client.auth_token == "Bearer dyn"
