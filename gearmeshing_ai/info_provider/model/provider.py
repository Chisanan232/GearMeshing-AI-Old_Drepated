"""Concrete model provider implementations.

This module contains the built-in model providers used by
open-source and local deployments, as well as composition helpers:

* :class:`HardcodedModelProvider` – in-process, dictionary-backed provider
  with a set of common model configurations.
* :class:`DatabaseModelProvider` – database-backed provider using agent_configs table.
* :class:`StackedModelProvider` – combines two providers with fallback
  semantics (typically database over hardcoded).
* :class:`HotReloadModelWrapper` – wraps another provider to call ``refresh``
  periodically in a thread-safe way.

Higher-level code is expected to access these via the
``gearmeshing_ai.info_provider.model`` facade.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

# Import ModelConfig for type hints
from gearmeshing_ai.core.models.config import ModelConfig

from .base import ModelProvider

# Minimal, builtin model configurations for basic/local usage. These are
# intentionally conservative and generic. In production deployments these
# are usually overridden by database configurations or custom providers.

_BUILTIN_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt4_default": ModelConfig(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=4096, top_p=0.9),
    "gpt4_creative": ModelConfig(provider="openai", model="gpt-4o", temperature=1.0, max_tokens=4096, top_p=1.0),
    "gpt4_precise": ModelConfig(provider="openai", model="gpt-4o", temperature=0.1, max_tokens=4096, top_p=0.5),
    "claude_sonnet": ModelConfig(
        provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=8192, top_p=0.9
    ),
    "claude_haiku": ModelConfig(
        provider="anthropic", model="claude-3-5-haiku-20241022", temperature=0.5, max_tokens=4096, top_p=0.8
    ),
    "gemini_pro": ModelConfig(provider="google", model="gemini-1.5-pro", temperature=0.7, max_tokens=2048, top_p=0.9),
    "gemini_flash": ModelConfig(
        provider="google", model="gemini-1.5-flash", temperature=0.3, max_tokens=1024, top_p=0.8
    ),
}


class HardcodedModelProvider(ModelProvider):
    """
    In-repo builtin model configurations for basic/local usage.

    This provider keeps a dictionary of model configurations in memory,
    keyed by configuration name. It serves as the baseline implementation
    for open-source usage and local development.

    Characteristics:
    - No tenant-specific content.
    - No external dependencies.
    - Fast, in-memory lookup.

    The :meth:`version` string is a lightweight identifier (e.g. "hardcoded-v1")
    used for tracking which model configuration set was active during a run.
    """

    def __init__(self, *, configs: Optional[Dict[str, ModelConfig]] = None, version_id: str = "hardcoded-v1") -> None:
        """
        Initialize the hardcoded model provider.

        Args:
            configs: Optional dictionary of model configs to override defaults.
            version_id: Identifier string for this config set version.
        """
        self._configs = configs or _BUILTIN_MODEL_CONFIGS
        self._version = version_id

    def get(self, name: str, tenant: Optional[str] = None) -> ModelConfig:  # noqa: ARG002
        """
        Retrieve a model configuration.

        Args:
            name: The model configuration key (e.g., 'gpt4_default').
            tenant: Ignored by this provider.

        Returns:
            ModelConfig: The model configuration.

        Raises:
            KeyError: If the model configuration name is not found.
        """
        try:
            return self._configs[name]
        except KeyError as exc:
            raise KeyError(f"model config not found: name={name!r}") from exc

    def version(self) -> str:
        """Return the version identifier of the builtin model configs."""
        return self._version

    def refresh(self) -> None:  # noqa: D401 - trivial no-op
        """Hardcoded provider has no external state to refresh (no-op)."""
        return None


class DatabaseModelProvider(ModelProvider):
    """
    Database-backed model configuration provider.

    This provider loads model configurations from the agent_configs table
    in the database, allowing for dynamic configuration management and
    multi-tenant support.

    Characteristics:
    - Tenant-specific configurations supported.
    - Dynamic configuration updates.
    - Requires database connection.
    """

    def __init__(self, db_session_factory, version_id: str = "database-v1") -> None:
        """
        Initialize the database model provider.

        Args:
            db_session_factory: Callable that returns a database session.
            version_id: Identifier string for this provider version.
        """
        self._db_session_factory = db_session_factory
        self._version = version_id

    def get(self, name: str, tenant: Optional[str] = None) -> ModelConfig:
        """
        Retrieve a model configuration from the database.

        Args:
            name: The role_name or configuration identifier.
            tenant: Optional tenant identifier for multi-tenant configs.

        Returns:
            ModelConfig: The model configuration from database.

        Raises:
            KeyError: If the configuration is not found.
        """
        with self._db_session_factory() as session:
            # Import here to avoid circular imports
            from gearmeshing_ai.server.models.agent_config import AgentConfig

            # Query for active configuration
            query = session.query(AgentConfig).filter(AgentConfig.role_name == name, AgentConfig.is_active == True)

            # Add tenant filter if specified
            if tenant is not None:
                query = query.filter(AgentConfig.tenant_id == tenant)
            else:
                # For non-tenant requests, prefer null tenant_id
                query = query.filter(AgentConfig.tenant_id.is_(None))  # type: ignore[union-attr]

            config = query.first()

            if not config:
                raise KeyError(f"model config not found: name={name!r}, tenant={tenant!r}")

            # Convert to ModelConfig
            return config.to_model_config()

    def version(self) -> str:
        """Return the version identifier of the database provider."""
        return self._version

    def refresh(self) -> None:  # noqa: D401 - no-op for database provider
        """Database provider queries fresh data on each call (no-op)."""
        return None


class StackedModelProvider(ModelProvider):
    """
    Chains two model providers with fallback semantics.

    This helper composes a primary (e.g., database) and a fallback (e.g., hardcoded)
    provider. It allows seamless degradation if the primary provider is missing keys
    or fails.

    Behavior:
    - ``get`` queries the primary first. If it raises ``KeyError``, the fallback is queried.
    - ``version`` returns a combined identifier ``"stacked:<primary>+<fallback>"``.
    - ``refresh`` refreshes both providers in order.
    """

    def __init__(self, primary: ModelProvider, fallback: ModelProvider) -> None:
        """
        Initialize the stacked model provider.

        Args:
            primary: The preferred provider to query first.
            fallback: The backup provider to query on cache miss.
        """
        self._primary = primary
        self._fallback = fallback

    def get(self, name: str, tenant: Optional[str] = None) -> ModelConfig:
        """
        Retrieve a model configuration, trying primary then fallback.

        Args:
            name: Model configuration key.
            tenant: Tenant identifier.

        Returns:
            ModelConfig: The model configuration.

        Raises:
            KeyError: If neither provider has the requested configuration.
        """
        try:
            return self._primary.get(name, tenant)
        except KeyError:
            return self._fallback.get(name, tenant)

    def version(self) -> str:
        """Return a composite version string reflecting both providers."""
        return f"stacked:{self._primary.version()}+{self._fallback.version()}"

    def refresh(self) -> None:
        """Refresh both the primary and fallback providers."""
        # Refresh both in order; callers that wrap this in a hot-reload wrapper
        # can still treat refresh as best-effort.
        self._primary.refresh()
        self._fallback.refresh()


class HotReloadModelWrapper(ModelProvider):
    """
    Lightweight, thread-safe wrapper that periodically refreshes a model provider.

    Designed for providers that fetch configurations from remote sources (HTTP, S3, DB).
    It ensures ``refresh()`` is called at most once per ``interval_seconds``, regardless
    of read concurrency.

    Attributes:
        inner: The underlying model provider instance.
        interval: Minimum seconds between refresh attempts.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        interval_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the hot-reload wrapper.

        Args:
            inner: The model provider to wrap and refresh.
            interval_seconds: Minimum time between refresh calls (default: 60s).
            logger: Optional logger for refresh errors (redacted).
        """
        self._inner = inner
        self._interval = max(interval_seconds, 0.0)
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._last_refresh: float = 0.0

    def _maybe_refresh(self) -> None:
        """
        Trigger a refresh on the underlying provider if the interval elapsed.

        This method uses double-checked locking to ensure thread safety and minimize contention.
        Errors during refresh are logged but swallowed to prevent read failures.
        """
        if self._interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_refresh < self._interval:
            return
        with self._lock:
            # Double-check under lock
            now = time.monotonic()
            if now - self._last_refresh < self._interval:
                return
            try:
                self._inner.refresh()
            except Exception as exc:  # pragma: no cover - defensive logging
                # Never log sensitive configuration data; only metadata.
                self._logger.warning(
                    "ModelProvider refresh failed; provider=%s version=%s error=%s",
                    type(self._inner).__name__,
                    self._safe_version(),
                    type(exc).__name__,
                )
            finally:
                self._last_refresh = time.monotonic()

    def _safe_version(self) -> str:
        """
        Return the inner provider's version, guarding against errors.

        Used for safe logging in case the inner provider is unstable.
        """
        try:
            return self._inner.version()
        except Exception:  # pragma: no cover - defensive
            return "<unknown>"

    def get(self, name: str, tenant: Optional[str] = None) -> ModelConfig:
        """Retrieve a model configuration, triggering a potential background refresh first."""
        self._maybe_refresh()
        return self._inner.get(name, tenant)

    def version(self) -> str:
        """Retrieve version, triggering a potential background refresh first."""
        self._maybe_refresh()
        return self._inner.version()

    def refresh(self) -> None:
        """
        Explicitly refresh the inner provider, bypassing throttling.

        Useful for administrative force-reloads. Resets the throttle timer.
        """
        try:
            self._inner.refresh()
        finally:
            self._last_refresh = time.monotonic()
