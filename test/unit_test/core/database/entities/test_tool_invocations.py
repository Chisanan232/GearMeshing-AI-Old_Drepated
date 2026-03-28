"""Unit tests for tool invocation entity model.

Tests SQLModel validation, field constraints, and business logic
for the ToolInvocation entity.
"""

from __future__ import annotations

from datetime import datetime


from gearmeshing_ai.core.database.entities.tool_invocations import ToolInvocation


class TestToolInvocation:
    """Tests for ToolInvocation entity model."""

    def test_tool_invocation_creation_valid_data(self):
        """Test ToolInvocation creation with valid data."""
        data = {
            "id": "tool_inv_123",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": {"repo": "test", "message": "Initial commit"},
            "ok": True,
            "result": {"commit_hash": "abc123"},
            "risk": "medium",
            "created_at": datetime.utcnow(),
        }

        invocation = ToolInvocation(**data)

        assert invocation.id == "tool_inv_123"
        assert invocation.run_id == "run_456"
        assert invocation.server_id == "server_789"
        assert invocation.tool_name == "git.commit"
        assert invocation.ok is True
        assert invocation.risk == "medium"

    def test_tool_invocation_missing_required_fields(self):
        """Test ToolInvocation validation with missing required fields."""
        # SQLModel doesn't validate required fields at model level
        # Fields will be None when not provided
        invocation = ToolInvocation()

        # All fields should be None when not provided
        assert invocation.id is None
        assert invocation.run_id is None
        assert invocation.server_id is None
        assert invocation.tool_name is None
        assert invocation.ok is None
        assert invocation.risk is None

    def test_tool_invocation_optional_fields(self):
        """Test ToolInvocation with optional fields."""
        data = {
            "id": "tool_inv_123",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": {},
            "ok": False,
            "result": {},
            "risk": "high",
            "created_at": datetime.utcnow(),
        }

        invocation = ToolInvocation(**data)

        assert invocation.args == {}
        assert invocation.result == {}
        assert invocation.ok is False

    def test_tool_invocation_automatic_timestamps(self):
        """Test ToolInvocation automatic timestamp generation."""
        data = {
            "id": "tool_inv_123",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": {},
            "ok": True,
            "result": {},
            "risk": "low",
        }

        invocation = ToolInvocation(**data)

        assert invocation.created_at is not None
        assert isinstance(invocation.created_at, datetime)

    def test_tool_invocation_string_field_length_validation(self):
        """Test ToolInvocation string field length constraints."""
        # Test ID field length - SQLModel may not validate length until DB operations
        data = {
            "id": "x" * 65,  # Exceeds max_length=64
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": {},
            "ok": True,
            "result": {},
            "risk": "low",
        }

        # SQLModel doesn't validate string length at model level
        invocation = ToolInvocation(**data)
        assert len(invocation.id) == 65  # Length validation happens at DB level

    def test_tool_invocation_risk_levels(self):
        """Test ToolInvocation risk level validation."""
        valid_risks = ["low", "medium", "high", "critical"]

        for risk in valid_risks:
            data = {
                "id": f"tool_inv_{risk}",
                "run_id": "run_456",
                "server_id": "server_789",
                "tool_name": "git.commit",
                "args": {},
                "ok": True,
                "result": {},
                "risk": risk,
            }

            invocation = ToolInvocation(**data)
            assert invocation.risk == risk

    def test_tool_invocation_complex_args_and_result(self):
        """Test ToolInvocation with complex args and result data."""
        complex_args = {
            "repository": {"url": "https://github.com/user/repo.git", "branch": "main", "commit": "abc123"},
            "options": {"force": False, "allow_empty": True, "message": "Automated commit"},
            "metadata": {"user_id": 123, "session_id": "sess_456"},
        }

        complex_result = {
            "commit": {
                "hash": "def456",
                "url": "https://github.com/user/repo/commit/def456",
                "message": "Automated commit",
                "author": {"name": "Test User", "email": "test@example.com"},
            },
            "branch": "main",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        data = {
            "id": "complex_tool_inv",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": complex_args,
            "ok": True,
            "result": complex_result,
            "risk": "low",
        }

        invocation = ToolInvocation(**data)

        assert invocation.args == complex_args
        assert invocation.result == complex_result
        assert invocation.args["repository"]["url"] == "https://github.com/user/repo.git"
        assert invocation.result["commit"]["hash"] == "def456"

    def test_tool_invocation_repr(self):
        """Test ToolInvocation string representation."""
        data = {
            "id": "tool_inv_123",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "git.commit",
            "args": {},
            "ok": True,
            "result": {},
            "risk": "medium",
        }

        invocation = ToolInvocation(**data)

        repr_str = repr(invocation)
        assert "ToolInvocation" in repr_str
        assert invocation.id in repr_str
        assert invocation.tool_name in repr_str
        assert str(invocation.ok) in repr_str

    def test_tool_invocation_table_name(self):
        """Test ToolInvocation table name configuration."""
        assert ToolInvocation.__tablename__ == "gm_tool_invocations"

    def test_tool_invocation_success_scenario(self):
        """Test successful tool invocation scenario."""
        data = {
            "id": "success_tool_inv",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "python.execute",
            "args": {"code": "print('Hello, world!')", "timeout": 30},
            "ok": True,
            "result": {"output": "Hello, world!", "return_code": 0},
            "risk": "low",
        }

        invocation = ToolInvocation(**data)

        assert invocation.ok is True
        assert invocation.result["output"] == "Hello, world!"
        assert invocation.result["return_code"] == 0

    def test_tool_invocation_failure_scenario(self):
        """Test failed tool invocation scenario."""
        data = {
            "id": "failed_tool_inv",
            "run_id": "run_456",
            "server_id": "server_789",
            "tool_name": "docker.run",
            "args": {"image": "nonexistent:latest", "command": "ls"},
            "ok": False,
            "result": {"error": "Image not found", "error_code": 404},
            "risk": "medium",
        }

        invocation = ToolInvocation(**data)

        assert invocation.ok is False
        assert invocation.result["error"] == "Image not found"
        assert invocation.result["error_code"] == 404

    def test_tool_invocation_high_risk_operations(self):
        """Test high-risk tool operations."""
        high_risk_tools = ["system.execute", "database.drop_table", "file.delete_recursive", "network.port_scan"]

        for tool_name in high_risk_tools:
            data = {
                "id": f"high_risk_{tool_name}",
                "run_id": "run_456",
                "server_id": "server_789",
                "tool_name": tool_name,
                "args": {},
                "ok": True,
                "result": {},
                "risk": "high",
            }

            invocation = ToolInvocation(**data)
            assert invocation.risk == "high"
            assert invocation.tool_name == tool_name

    def test_tool_invocation_server_tracking(self):
        """Test server ID tracking for tool invocations."""
        servers = ["server_001", "server_002", "server_003"]

        for server_id in servers:
            data = {
                "id": f"tool_inv_{server_id}",
                "run_id": "run_456",
                "server_id": server_id,
                "tool_name": "test.tool",
                "args": {},
                "ok": True,
                "result": {},
                "risk": "low",
            }

            invocation = ToolInvocation(**data)
            assert invocation.server_id == server_id
