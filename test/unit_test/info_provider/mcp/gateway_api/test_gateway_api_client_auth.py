from __future__ import annotations

import importlib
import subprocess
from typing import Any, Dict

import pytest

from gearmeshing_ai.info_provider.mcp.gateway_api.client import GatewayApiClient
from gearmeshing_ai.info_provider.mcp.gateway_api.errors import GatewayApiError


def test_generate_bearer_token_success_sets_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_check_output(cmd, env=None, timeout=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = env or {}
        captured["timeout"] = timeout
        assert captured["env"].get("MCPGATEWAY_JWT_SECRET") == "abc123"
        return b"jwt-token"

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        # Simulate mcpgateway presence
        if name in ("mcpgateway", "mcpgateway.utils.create_jwt_token"):
            return object()
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
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

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name in ("mcpgateway", "mcpgateway.utils.create_jwt_token"):
            return object()
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    token = GatewayApiClient.generate_bearer_token(jwt_secret_key="k", extra_env={"FOO": "bar"})
    assert token == "Bearer t"


def test_init_auto_bearer_sets_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_output(cmd, env=None, timeout=None):  # type: ignore[no-untyped-def]
        return b"abc"

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name in ("mcpgateway", "mcpgateway.utils.create_jwt_token"):
            return object()
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
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


def test_init_auto_bearer_failure_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def failing_check_output(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("no jwt")

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name in ("mcpgateway", "mcpgateway.utils.create_jwt_token"):
            return object()
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(subprocess, "check_output", failing_check_output)
    caplog.clear()
    caplog.set_level("WARNING")
    client = GatewayApiClient("http://mock", auto_bearer=True, jwt_secret_key="s")
    # Should not raise; should log warning and leave auth_token unset
    assert client.auth_token is None
    assert any("auto_bearer failed" in rec.message for rec in caplog.records)


def test_token_provider_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    def provider():  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    client = GatewayApiClient("http://mock", token_provider=provider)
    caplog.clear()
    caplog.set_level("WARNING")
    client._ensure_token()
    assert client.auth_token is None
    assert any("token_provider failed" in rec.message for rec in caplog.records)


def test_generate_bearer_token_raises_gateway_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_check_output(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("subprocess error")

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name in ("mcpgateway", "mcpgateway.utils.create_jwt_token"):
            return object()
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(subprocess, "check_output", failing_check_output)
    with pytest.raises(GatewayApiError):
        GatewayApiClient.generate_bearer_token(jwt_secret_key="k")


def test_generate_bearer_token_raises_if_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(GatewayApiError):
        GatewayApiClient.generate_bearer_token(jwt_secret_key="k")
