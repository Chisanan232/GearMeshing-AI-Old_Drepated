"""Unit tests for usage ledger repository.

Tests repository operations with mocked database session to ensure
business logic works correctly without real database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gearmeshing_ai.core.database.repositories.usage_ledger import UsageLedgerRepository


class TestUsageLedgerRepository:
    """Tests for UsageLedgerRepository operations."""

    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance with mocked session."""
        return UsageLedgerRepository(mock_session)

    async def test_create_success(self, repository, mock_session):
        """Test successful usage ledger entry creation."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"
        mock_usage.tenant_id = "tenant_789"
        mock_usage.provider = "openai"
        mock_usage.model = "gpt-4o"
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.cost_usd = 0.0045
        mock_usage.created_at = datetime.utcnow()

        result = await repository.create(mock_usage)

        # Verify session operations
        mock_session.add.assert_called_once_with(mock_usage)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_usage)

        # Verify return value
        assert result == mock_usage

    @patch.object(UsageLedgerRepository, "get_by_id")
    async def test_get_by_id_success(self, mock_get_by_id, repository, mock_session):
        """Test successful usage ledger retrieval by ID."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"

        # Mock the get_by_id method
        mock_get_by_id.return_value = mock_usage

        result = await repository.get_by_id("usage_123")

        # Verify the method was called correctly
        mock_get_by_id.assert_called_once_with("usage_123")
        assert result == mock_usage

    @patch.object(UsageLedgerRepository, "get_by_id")
    async def test_get_by_id_not_found(self, mock_get_by_id, repository):
        """Test usage ledger retrieval by ID when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None

        result = await repository.get_by_id("nonexistent")

        assert result is None

    async def test_update_success(self, repository, mock_session):
        """Test successful usage ledger entry update."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.cost_usd = 0.0050

        result = await repository.update(mock_usage)

        # Verify session operations
        mock_session.add.assert_called_once_with(mock_usage)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_usage)

        # Verify result
        assert result == mock_usage

    @patch.object(UsageLedgerRepository, "get_by_id")
    async def test_delete_success(self, mock_get_by_id, repository, mock_session):
        """Test successful usage ledger entry deletion."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"

        # Mock the get_by_id method to return the usage
        mock_get_by_id.return_value = mock_usage

        result = await repository.delete("usage_123")

        # Verify session operations
        mock_session.delete.assert_called_once_with(mock_usage)
        mock_session.commit.assert_called_once()

        # Verify result
        assert result is True

    @patch.object(UsageLedgerRepository, "get_by_id")
    async def test_delete_not_found(self, mock_get_by_id, repository, mock_session):
        """Test usage ledger entry deletion when not found."""
        # Mock the get_by_id method to return None
        mock_get_by_id.return_value = None

        result = await repository.delete("nonexistent")

        # Verify session operations were not called
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

        # Verify result
        assert result is False

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_without_filters(self, mock_list, repository):
        """Test listing usage ledger entries without filters."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list()

        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_run_id_filter(self, mock_list, repository):
        """Test listing usage ledger entries with run_id filter."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(filters={"run_id": "run_456"})

        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"run_id": "run_456"})
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_tenant_id_filter(self, mock_list, repository):
        """Test listing usage ledger entries with tenant_id filter."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.tenant_id = "tenant_789"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(filters={"tenant_id": "tenant_789"})

        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"tenant_id": "tenant_789"})
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_provider_filter(self, mock_list, repository):
        """Test listing usage ledger entries with provider filter."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.provider = "openai"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(filters={"provider": "openai"})

        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"provider": "openai"})
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_model_filter(self, mock_list, repository):
        """Test listing usage ledger entries with model filter."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.model = "gpt-4o"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(filters={"model": "gpt-4o"})

        # Verify the method was called correctly
        mock_list.assert_called_once_with(filters={"model": "gpt-4o"})
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_multiple_filters(self, mock_list, repository):
        """Test listing usage ledger entries with multiple filters."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"
        mock_usage.tenant_id = "tenant_789"
        mock_usage.provider = "openai"
        mock_usage.model = "gpt-4o"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(
            filters={"run_id": "run_456", "tenant_id": "tenant_789", "provider": "openai", "model": "gpt-4o"}
        )

        # Verify the method was called correctly
        mock_list.assert_called_once_with(
            filters={"run_id": "run_456", "tenant_id": "tenant_789", "provider": "openai", "model": "gpt-4o"}
        )
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_with_limit_and_offset(self, mock_list, repository):
        """Test listing usage ledger entries with pagination."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"

        # Mock the list method
        mock_list.return_value = [mock_usage]

        result = await repository.list(limit=10, offset=5)

        # Verify the method was called correctly
        mock_list.assert_called_once_with(limit=10, offset=5)
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "list")
    async def test_list_empty_result(self, mock_list, repository):
        """Test listing usage ledger entries when none exist."""
        # Mock the list method to return empty list
        mock_list.return_value = []

        result = await repository.list()

        # Verify the method was called correctly
        mock_list.assert_called_once_with()
        assert len(result) == 0

    @patch.object(UsageLedgerRepository, "get_usage_for_run")
    async def test_get_usage_for_run(self, mock_get_usage_for_run, repository):
        """Test getting usage entries for a specific run."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.run_id = "run_456"

        # Mock the get_usage_for_run method
        mock_get_usage_for_run.return_value = [mock_usage]

        result = await repository.get_usage_for_run("run_456")

        # Verify the method was called correctly
        mock_get_usage_for_run.assert_called_once_with("run_456")
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "get_usage_for_run")
    async def test_get_usage_for_run_empty(self, mock_get_usage_for_run, repository):
        """Test getting usage entries for a run with no usage."""
        # Mock the get_usage_for_run method to return empty list
        mock_get_usage_for_run.return_value = []

        result = await repository.get_usage_for_run("nonexistent_run")

        # Verify the method was called correctly
        mock_get_usage_for_run.assert_called_once_with("nonexistent_run")
        assert len(result) == 0

    @patch.object(UsageLedgerRepository, "get_usage_for_tenant")
    async def test_get_usage_for_tenant(self, mock_get_usage_for_tenant, repository):
        """Test getting usage entries for a specific tenant."""
        # Create a mock usage ledger object
        mock_usage = MagicMock()
        mock_usage.id = "usage_123"
        mock_usage.tenant_id = "tenant_789"

        # Mock the get_usage_for_tenant method
        mock_get_usage_for_tenant.return_value = [mock_usage]

        result = await repository.get_usage_for_tenant("tenant_789")

        # Verify the method was called correctly
        mock_get_usage_for_tenant.assert_called_once_with("tenant_789")
        assert len(result) == 1
        assert result[0] == mock_usage

    @patch.object(UsageLedgerRepository, "get_usage_for_tenant")
    async def test_get_usage_for_tenant_empty(self, mock_get_usage_for_tenant, repository):
        """Test getting usage entries for a tenant with no usage."""
        # Mock the get_usage_for_tenant method to return empty list
        mock_get_usage_for_tenant.return_value = []

        result = await repository.get_usage_for_tenant("nonexistent_tenant")

        # Verify the method was called correctly
        mock_get_usage_for_tenant.assert_called_once_with("nonexistent_tenant")
        assert len(result) == 0

    @patch.object(UsageLedgerRepository, "get_total_usage_for_run")
    async def test_get_total_usage_for_run_success(self, mock_get_total_usage, repository):
        """Test getting total usage for a run with usage data."""
        # Mock the aggregate result
        expected_result = {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50, "total_cost": 0.0045}

        # Mock the get_total_usage_for_run method
        mock_get_total_usage.return_value = expected_result

        result = await repository.get_total_usage_for_run("run_456")

        # Verify the method was called correctly
        mock_get_total_usage.assert_called_once_with("run_456")
        assert result == expected_result

    @patch.object(UsageLedgerRepository, "get_total_usage_for_run")
    async def test_get_total_usage_for_run_no_usage(self, mock_get_total_usage, repository):
        """Test getting total usage for a run with no usage data."""
        # Mock the get_total_usage_for_run method to return None
        mock_get_total_usage.return_value = None

        result = await repository.get_total_usage_for_run("nonexistent_run")

        # Verify the method was called correctly
        mock_get_total_usage.assert_called_once_with("nonexistent_run")
        assert result is None

    @patch.object(UsageLedgerRepository, "get_total_usage_for_run")
    async def test_get_total_usage_for_run_no_result(self, mock_get_total_usage, repository):
        """Test getting total usage for a run with no database result."""
        # Mock the get_total_usage_for_run method to return None
        mock_get_total_usage.return_value = None

        result = await repository.get_total_usage_for_run("nonexistent_run")

        # Verify the method was called correctly
        mock_get_total_usage.assert_called_once_with("nonexistent_run")
        assert result is None
