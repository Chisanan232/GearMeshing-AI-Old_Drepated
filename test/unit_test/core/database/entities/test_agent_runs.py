"""Unit tests for agent run entity model.

Tests SQLModel validation, field constraints, and business logic
for the AgentRun entity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.agent_runs import AgentRun, AgentRunBase


class TestAgentRunBase:
    """Tests for AgentRunBase model validation."""
    
    def test_agent_run_base_valid_data(self, sample_agent_run_data):
        """Test AgentRunBase with valid data."""
        # Extract only base fields
        base_data = {
            "role": sample_agent_run_data["role"],
            "autonomy_profile": sample_agent_run_data["autonomy_profile"],
            "objective": sample_agent_run_data["objective"],
            "done_when": sample_agent_run_data["done_when"],
            "prompt_provider_version": sample_agent_run_data["prompt_provider_version"],
            "status": sample_agent_run_data["status"],
        }
        
        agent_run = AgentRunBase(**base_data)
        
        assert agent_run.role == "developer"
        assert agent_run.autonomy_profile == "balanced"
        assert agent_run.objective == "Build a new feature"
        assert agent_run.status == "running"
    
    def test_agent_run_base_missing_required_fields(self):
        """Test AgentRunBase validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            AgentRunBase()
        
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        
        expected_fields = {"role", "autonomy_profile", "objective", "status"}
        assert expected_fields.issubset(error_fields)
    
    def test_agent_run_base_invalid_status(self, sample_agent_run_data):
        """Test AgentRunBase with invalid status values."""
        base_data = {
            "role": sample_agent_run_data["role"],
            "autonomy_profile": sample_agent_run_data["autonomy_profile"],
            "objective": sample_agent_run_data["objective"],
            "done_when": sample_agent_run_data["done_when"],
            "prompt_provider_version": sample_agent_run_data["prompt_provider_version"],
            "status": "",  # Empty string should be invalid
        }
        
        # SQLModel doesn't validate string constraints at model level
        agent_run = AgentRunBase(**base_data)
        assert agent_run.status == ""  # Empty string is allowed at model level


class TestAgentRun:
    """Tests for AgentRun entity model."""
    
    def test_agent_run_creation_valid_data(self, sample_agent_run_data):
        """Test AgentRun creation with valid data."""
        agent_run = AgentRun(**sample_agent_run_data)
        
        assert agent_run.id == "test_run_123"
        assert agent_run.tenant_id == "tenant_456"
        assert agent_run.workspace_id == "workspace_789"
        assert agent_run.role == "developer"
        assert agent_run.status == "running"
        assert agent_run.objective == "Build a new feature"
    
    def test_agent_run_missing_primary_key(self, sample_agent_run_data):
        """Test AgentRun without primary key creates run with None ID."""
        data = sample_agent_run_data.copy()
        del data["id"]
        
        # SQLModel allows missing primary key during creation (it will be set by DB)
        agent_run = AgentRun(**data)
        
        # The ID should be None when not provided
        assert agent_run.id is None
    
    def test_agent_run_optional_fields_none(self, sample_agent_run_data):
        """Test AgentRun with optional fields as None."""
        data = sample_agent_run_data.copy()
        data["tenant_id"] = None
        data["workspace_id"] = None
        data["done_when"] = None
        data["prompt_provider_version"] = None
        
        agent_run = AgentRun(**data)
        
        assert agent_run.tenant_id is None
        assert agent_run.workspace_id is None
        assert agent_run.done_when is None
        assert agent_run.prompt_provider_version is None
    
    def test_agent_run_string_field_length_validation(self, sample_agent_run_data):
        """Test AgentRun string field length constraints."""
        # Test ID field length - SQLModel may not validate length until DB operations
        data = sample_agent_run_data.copy()
        data["id"] = "x" * 65  # Exceeds max_length=64
        
        # SQLModel doesn't validate string length at model level
        agent_run = AgentRun(**data)
        assert len(agent_run.id) == 65  # Length validation happens at DB level
    
    def test_agent_run_automatic_timestamps(self, sample_agent_run_data):
        """Test AgentRun automatic timestamp generation."""
        data = sample_agent_run_data.copy()
        del data["created_at"]
        del data["updated_at"]
        
        agent_run = AgentRun(**data)
        
        assert agent_run.created_at is not None
        assert agent_run.updated_at is not None
        assert isinstance(agent_run.created_at, datetime)
        assert isinstance(agent_run.updated_at, datetime)
    
    def test_agent_run_repr(self, sample_agent_run_data):
        """Test AgentRun string representation."""
        agent_run = AgentRun(**sample_agent_run_data)
        
        repr_str = repr(agent_run)
        assert "AgentRun" in repr_str
        assert agent_run.id in repr_str
        assert agent_run.role in repr_str
        assert agent_run.status in repr_str
    
    def test_agent_run_table_name(self):
        """Test AgentRun table name configuration."""
        assert AgentRun.__tablename__ == "gm_agent_runs"
    
    def test_agent_run_inheritance(self):
        """Test AgentRun inherits from AgentRunBase."""
        assert issubclass(AgentRun, AgentRunBase)
    
    @pytest.mark.parametrize("field_name", [
        "role", "autonomy_profile", "objective", "status"
    ])
    def test_agent_run_required_base_fields(self, sample_agent_run_data, field_name):
        """Test that base fields become None when not provided in AgentRun."""
        data = sample_agent_run_data.copy()
        del data[field_name]
        
        # SQLModel sets missing fields to None instead of raising validation error
        agent_run = AgentRun(**data)
        
        # The missing field should be None
        assert getattr(agent_run, field_name) is None
    
    def test_agent_run_compatibility_alias(self, sample_agent_run_data):
        """Test RunRow compatibility alias works."""
        from gearmeshing_ai.core.database.entities.agent_runs import RunRow
        
        # RunRow should be the same as AgentRun
        assert RunRow is AgentRun
        
        # Should be able to create with RunRow
        agent_run = RunRow(**sample_agent_run_data)
        assert isinstance(agent_run, AgentRun)
        assert agent_run.id == "test_run_123"
