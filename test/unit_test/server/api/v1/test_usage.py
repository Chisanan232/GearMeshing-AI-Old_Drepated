"""
Unit tests for Usage Statistics API endpoints.

Tests cover:
- Retrieving aggregated usage statistics
- Filtering by date range
- Calculating total tokens and costs
- Handling empty usage data

These tests use direct function calls to ensure proper coverage detection of async code.
See TestDirectFunctionCalls class documentation for why direct calls are necessary.
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

from gearmeshing_ai.agent_core.schemas.domain import UsageLedgerEntry

pytestmark = pytest.mark.asyncio


class TestDirectFunctionCalls:
    """Direct function call tests to ensure proper coverage detection of async code.

    IMPORTANT: These tests call the endpoint functions directly with mocked orchestrator,
    bypassing the HTTP layer so coverage.py can properly track execution.

    WHY DIRECT CALLS ARE NECESSARY FOR COVERAGE:
    ============================================
    When tests use AsyncClient (HTTP layer), the code execution flow goes through:
    1. HTTP request serialization
    2. FastAPI routing and dependency injection
    3. Async context managers and middleware
    4. The actual endpoint function
    5. Response serialization

    Coverage.py struggles to properly instrument and track code execution in this
    flow because:
    - Async context managers can obscure the actual code paths
    - Exception handling in FastAPI's exception handlers may not be tracked
    - The HTTP layer adds abstraction that coverage tools can't fully penetrate
    - Orchestrator async operations may not be properly detected when called through HTTP

    SOLUTION: Direct function calls with mocked orchestrator allow coverage.py to:
    - Directly instrument the endpoint function code
    - Track all await statements and async operations
    - Properly detect aggregation logic
    - Verify that specific lines like orchestrator calls execute

    LINES THAT REQUIRE DIRECT CALLS:
    - Line 35: await orchestrator.list_usage(tenant_id=tenant_id, from_date=from_date, to_date=to_date)
    - Lines 38-39: Aggregation of total_tokens and total_cost
    - Lines 41-47: Return statement with aggregated data
    """

    async def test_get_usage_with_entries_direct_call(self):
        """Test get usage endpoint directly - covers lines 35-47.

        COVERAGE TARGET: Lines 35-47 in usage.py
            entries = await orchestrator.list_usage(...)
            total_tokens = sum(e.total_tokens for e in entries)
            total_cost = sum(e.cost_usd or 0.0 for e in entries)
            return {...}

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect await orchestrator.list_usage() execution
        - HTTP layer cannot properly detect aggregation logic
        - Direct function call allows coverage.py to instrument the actual await statement

        VERIFICATION:
        - result is not None: Proves the orchestrator call was awaited
        - result.total_tokens == 1000: Proves aggregation was calculated
        - result.total_cost_usd == 10.0: Proves cost aggregation was calculated
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_entries = [
            UsageLedgerEntry(
                id="entry-1",
                run_id="run-1",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost_usd=3.0
            ),
            UsageLedgerEntry(
                id="entry-2",
                run_id="run-2",
                prompt_tokens=200,
                completion_tokens=500,
                total_tokens=700,
                cost_usd=7.0
            )
        ]
        mock_orchestrator.list_usage = AsyncMock(return_value=mock_entries)

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=None,
            to_date=None
        )

        # Verify
        assert result is not None
        assert result["tenant_id"] == "test-tenant"
        assert result["total_tokens"] == 1000
        assert result["total_cost_usd"] == 10.0
        assert result["entries_count"] == 2
        mock_orchestrator.list_usage.assert_called_once_with(
            tenant_id="test-tenant", from_date=None, to_date=None
        )

    async def test_get_usage_empty_entries_direct_call(self):
        """Test get usage with no entries - covers lines 35-47.

        COVERAGE TARGET: Lines 35-47 in usage.py (with empty entries)
            entries = await orchestrator.list_usage(...)
            total_tokens = sum(e.total_tokens for e in entries)  # 0
            total_cost = sum(e.cost_usd or 0.0 for e in entries)  # 0.0
            return {...}

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect aggregation with empty list
        - Direct function call allows coverage.py to track empty aggregation path

        VERIFICATION:
        - result.total_tokens == 0: Proves empty aggregation was calculated
        - result.total_cost_usd == 0.0: Proves zero cost aggregation
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.list_usage = AsyncMock(return_value=[])

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=None,
            to_date=None
        )

        # Verify
        assert result is not None
        assert result["tenant_id"] == "test-tenant"
        assert result["total_tokens"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["entries_count"] == 0

    async def test_get_usage_with_date_range_direct_call(self):
        """Test get usage with date range - covers lines 35-47.

        COVERAGE TARGET: Lines 35-47 in usage.py (with date parameters)
            entries = await orchestrator.list_usage(
                tenant_id=tenant_id,
                from_date=from_date,
                to_date=to_date
            )

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect date parameter passing
        - Direct function call allows coverage.py to track date parameter path

        VERIFICATION:
        - orchestrator called with correct dates: Proves date parameters were passed
        - result contains aggregated data: Proves aggregation worked with date filter
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        from_date = datetime(2025, 1, 1)
        to_date = datetime(2025, 1, 31)
        
        mock_entries = [
            UsageLedgerEntry(
                id="entry-1",
                run_id="run-1",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost_usd=3.0
            )
        ]
        mock_orchestrator.list_usage = AsyncMock(return_value=mock_entries)

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=from_date,
            to_date=to_date
        )

        # Verify
        assert result is not None
        assert result["period"]["from"] == from_date
        assert result["period"]["to"] == to_date
        assert result["total_tokens"] == 300
        mock_orchestrator.list_usage.assert_called_once_with(
            tenant_id="test-tenant", from_date=from_date, to_date=to_date
        )

    async def test_get_usage_with_null_costs_direct_call(self):
        """Test get usage with null cost values - covers lines 38-39.

        COVERAGE TARGET: Lines 38-39 in usage.py
            total_cost = sum(e.cost_usd or 0.0 for e in entries)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect null cost handling
        - Direct function call allows coverage.py to track null value path

        VERIFICATION:
        - result.total_cost_usd == 5.0: Proves null costs were treated as 0.0
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_entries = [
            UsageLedgerEntry(
                id="entry-1",
                run_id="run-1",
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost_usd=5.0
            ),
            UsageLedgerEntry(
                id="entry-2",
                run_id="run-2",
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150,
                cost_usd=None  # Null cost
            )
        ]
        mock_orchestrator.list_usage = AsyncMock(return_value=mock_entries)

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=None,
            to_date=None
        )

        # Verify
        assert result is not None
        assert result["total_tokens"] == 450
        assert result["total_cost_usd"] == 5.0  # Only the non-null cost
        assert result["entries_count"] == 2

    async def test_get_usage_large_numbers_direct_call(self):
        """Test get usage with large token and cost numbers - covers lines 38-39.

        COVERAGE TARGET: Lines 38-39 in usage.py (with large numbers)
            total_tokens = sum(e.total_tokens for e in entries)
            total_cost = sum(e.cost_usd or 0.0 for e in entries)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect large number aggregation
        - Direct function call allows coverage.py to track large number path

        VERIFICATION:
        - result.total_tokens == 1000000: Proves large number aggregation
        - result.total_cost_usd == 10000.0: Proves large cost aggregation
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_entries = [
            UsageLedgerEntry(
                id="entry-1",
                run_id="run-1",
                prompt_tokens=300000,
                completion_tokens=200000,
                total_tokens=500000,
                cost_usd=5000.0
            ),
            UsageLedgerEntry(
                id="entry-2",
                run_id="run-2",
                prompt_tokens=300000,
                completion_tokens=200000,
                total_tokens=500000,
                cost_usd=5000.0
            )
        ]
        mock_orchestrator.list_usage = AsyncMock(return_value=mock_entries)

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=None,
            to_date=None
        )

        # Verify
        assert result is not None
        assert result["total_tokens"] == 1000000
        assert result["total_cost_usd"] == 10000.0
        assert result["entries_count"] == 2

    async def test_get_usage_single_entry_direct_call(self):
        """Test get usage with single entry - covers lines 35-47.

        COVERAGE TARGET: Lines 35-47 in usage.py (with single entry)
            entries = await orchestrator.list_usage(...)
            total_tokens = sum(e.total_tokens for e in entries)
            total_cost = sum(e.cost_usd or 0.0 for e in entries)

        WHY DIRECT CALL IS NEEDED:
        - HTTP layer cannot properly detect single entry aggregation
        - Direct function call allows coverage.py to track single entry path

        VERIFICATION:
        - result.entries_count == 1: Proves single entry was processed
        - result.total_tokens == 500: Proves single entry aggregation
        """
        from gearmeshing_ai.server.api.v1 import usage

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_entries = [
            UsageLedgerEntry(
                id="entry-1",
                run_id="run-1",
                prompt_tokens=200,
                completion_tokens=300,
                total_tokens=500,
                cost_usd=5.0
            )
        ]
        mock_orchestrator.list_usage = AsyncMock(return_value=mock_entries)

        # Call endpoint directly
        result = await usage.get_usage(
            mock_orchestrator,
            tenant_id="test-tenant",
            from_date=None,
            to_date=None
        )

        # Verify
        assert result is not None
        assert result["entries_count"] == 1
        assert result["total_tokens"] == 500
        assert result["total_cost_usd"] == 5.0
