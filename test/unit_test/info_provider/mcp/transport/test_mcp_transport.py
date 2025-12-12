from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Tuple

import pytest

from gearmeshing_ai.info_provider.mcp.transport import mcp as m_mod


class _FakeClientSession:
    def __init__(self, read_stream: Any, write_stream: Any) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False
        self.entered = False
        self.exited = False
        self.exit_exc: Tuple[type | None, BaseException | None, Any] | None = None

    async def __aenter__(self) -> "_FakeClientSession":
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True
        self.exit_exc = (exc_type, exc, tb)
        return None

    async def initialize(self) -> None:
        self.initialized = True


def _fake_streamablehttp_client_factory(
    state: dict,
) -> Callable[[str], AsyncIterator[Tuple[Any, Any, Callable[[], None]]]]:
    @asynccontextmanager
    async def _fake_ctx(endpoint_url: str):  # noqa: ARG001
        state["stream_entered"] = True
        read_stream = object()
        write_stream = object()

        def _close() -> None:
            state["close_called"] = True

        try:
            yield (read_stream, write_stream, _close)
        finally:
            state["stream_exited"] = True

    return _fake_ctx  # type: ignore[return-value]


def _fake_sse_client_factory(state: dict) -> Callable[[str], AsyncIterator[Tuple[Any, Any]]]:
    @asynccontextmanager
    async def _fake_ctx(endpoint_url: str):  # noqa: ARG001
        state["sse_entered"] = True
        read_stream = object()
        write_stream = object()
        try:
            yield (read_stream, write_stream)
        finally:
            state["sse_exited"] = True

    return _fake_ctx  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_streamable_http_transport_session_initializes_and_yields(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict = {}

    # Patch the client factories and ClientSession
    monkeypatch.setattr(m_mod, "streamablehttp_client", _fake_streamablehttp_client_factory(state), raising=True)
    monkeypatch.setattr(m_mod, "ClientSession", _FakeClientSession, raising=True)

    transport = m_mod.StreamableHttpMCPTransport()

    async with transport.session("http://mock/mcp") as session:
        assert isinstance(session, _FakeClientSession)
        assert session.entered is True
        assert session.initialized is True
        # Ensure inner factory was entered
        assert state.get("stream_entered") is True

    # Exits should have been called
    assert state.get("stream_exited") is True
    assert session.exited is True


@pytest.mark.asyncio
async def test_sse_transport_session_initializes_and_yields(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict = {}

    monkeypatch.setattr(m_mod, "sse_client", _fake_sse_client_factory(state), raising=True)
    monkeypatch.setattr(m_mod, "ClientSession", _FakeClientSession, raising=True)

    transport = m_mod.SseMCPTransport()

    async with transport.session("http://mock/sse/sse") as session:
        assert isinstance(session, _FakeClientSession)
        assert session.entered is True
        assert session.initialized is True
        assert state.get("sse_entered") is True

    assert state.get("sse_exited") is True
    assert session.exited is True


@pytest.mark.asyncio
async def test_streamable_http_transport_propagates_initialize_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict = {}

    class _FailingClientSession(_FakeClientSession):
        async def initialize(self) -> None:
            raise RuntimeError("init-failed")

    monkeypatch.setattr(m_mod, "streamablehttp_client", _fake_streamablehttp_client_factory(state), raising=True)
    monkeypatch.setattr(m_mod, "ClientSession", _FailingClientSession, raising=True)

    transport = m_mod.StreamableHttpMCPTransport()

    with pytest.raises(RuntimeError, match="init-failed"):
        async with transport.session("http://mock/mcp"):
            pass

    # Even on error, inner context should exit and client session aexit should be called
    assert state.get("stream_exited") is True


@pytest.mark.asyncio
async def test_sse_transport_propagates_aenter_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate failure entering ClientSession
    state: dict = {}

    class _EnterFailClientSession(_FakeClientSession):
        async def __aenter__(self) -> "_FakeClientSession":
            raise ValueError("enter-failed")

    monkeypatch.setattr(m_mod, "sse_client", _fake_sse_client_factory(state), raising=True)
    monkeypatch.setattr(m_mod, "ClientSession", _EnterFailClientSession, raising=True)

    transport = m_mod.SseMCPTransport()

    with pytest.raises(ValueError, match="enter-failed"):
        async with transport.session("http://mock/sse/sse"):
            pass

    assert state.get("sse_exited") is True
