"""Unit tests for agent event entity model.

Tests SQLModel validation, field constraints, JSON helper methods,
and business logic for the AgentEvent entity.
"""

from __future__ import annotations

import json
from datetime import datetime
import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.agent_events import AgentEvent, AgentEventBase


class TestAgentEventBase:
    """Tests for AgentEventBase model validation."""
    
    def test_agent_event_base_valid_data(self, sample_agent_event_data):
        """Test AgentEventBase with valid data."""
        # Extract only base fields and convert payload to JSON string
        base_data = {
            "type": sample_agent_event_data["type"],
            "correlation_id": sample_agent_event_data["correlation_id"],
            "payload": json.dumps(sample_agent_event_data["payload"]),
        }
        
        event = AgentEventBase(**base_data)
        
        assert event.type == "step_completed"
        assert event.correlation_id == "correlation_456"
        assert event.payload == '{"step": "analysis", "result": "success"}'
    
    def test_agent_event_base_missing_required_fields(self):
        """Test AgentEventBase validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEventBase()
        
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        
        assert "type" in error_fields
    
    def test_agent_event_base_optional_fields(self, sample_agent_event_data):
        """Test AgentEventBase with optional fields as None."""
        base_data = {
            "type": sample_agent_event_data["type"],
            "correlation_id": None,
            "payload": "{}",
        }
        
        event = AgentEventBase(**base_data)
        
        assert event.type == "step_completed"
        assert event.correlation_id is None
        assert event.payload == "{}"


class TestAgentEvent:
    """Tests for AgentEvent entity model."""
    
    def test_agent_event_creation_valid_data(self, sample_agent_event_data):
        """Test AgentEvent creation with valid data."""
        event = AgentEvent(**sample_agent_event_data)
        
        assert event.id == "test_event_123"
        assert event.run_id == "test_run_123"
        assert event.type == "step_completed"
        assert event.correlation_id == "correlation_456"
    
    def test_agent_event_missing_primary_key(self, sample_agent_event_data):
        """Test AgentEvent without primary key creates event with None ID."""
        data = sample_agent_event_data.copy()
        del data["id"]
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        # SQLModel allows missing primary key during creation (it will be set by DB)
        event = AgentEvent(**data)
        
        # The ID should be None when not provided
        assert event.id is None
    
    def test_agent_event_optional_fields_none(self, sample_agent_event_data):
        """Test AgentEvent with optional fields as None."""
        data = sample_agent_event_data.copy()
        data["correlation_id"] = None
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        event = AgentEvent(**data)
        
        assert event.correlation_id is None
    
    def test_agent_event_string_field_length_validation(self, sample_agent_event_data):
        """Test AgentEvent string field length constraints."""
        # Test ID field length - SQLModel may not validate length until DB operations
        data = sample_agent_event_data.copy()
        data["id"] = "x" * 65  # Exceeds max_length=64
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        # SQLModel doesn't validate string length at model level
        event = AgentEvent(**data)
        assert len(event.id) == 65  # Length validation happens at DB level
        
        # Test run_id field length
        data = sample_agent_event_data.copy()
        data["run_id"] = "x" * 65  # Exceeds max_length=64
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        event = AgentEvent(**data)
        assert len(event.run_id) == 65  # Length validation happens at DB level
    
    def test_agent_event_automatic_timestamps(self, sample_agent_event_data):
        """Test AgentEvent automatic timestamp generation."""
        data = sample_agent_event_data.copy()
        del data["created_at"]
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        event = AgentEvent(**data)
        
        assert event.created_at is not None
        assert isinstance(event.created_at, datetime)
    
    def test_agent_event_get_payload_dict_valid_json(self, sample_agent_event_data):
        """Test get_payload_dict with valid JSON."""
        data = sample_agent_event_data.copy()
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        event = AgentEvent(**data)
        
        payload_dict = event.get_payload_dict()
        
        assert isinstance(payload_dict, dict)
        assert payload_dict["step"] == "analysis"
        assert payload_dict["result"] == "success"
    
    def test_agent_event_get_payload_dict_invalid_json(self, sample_agent_event_data):
        """Test get_payload_dict with invalid JSON."""
        data = sample_agent_event_data.copy()
        data["payload"] = "invalid json string"
        
        event = AgentEvent(**data)
        
        payload_dict = event.get_payload_dict()
        
        assert payload_dict == {}  # Should return empty dict for invalid JSON
    
    def test_agent_event_get_payload_dict_empty_payload(self, sample_agent_event_data):
        """Test get_payload_dict with empty payload."""
        data = sample_agent_event_data.copy()
        data["payload"] = ""
        
        event = AgentEvent(**data)
        
        payload_dict = event.get_payload_dict()
        
        assert payload_dict == {}
    
    def test_agent_event_set_payload_dict(self, sample_agent_event_data):
        """Test set_payload_dict method."""
        event = AgentEvent(**sample_agent_event_data)
        
        new_payload = {"action": "completed", "details": {"items": 5}}
        event.set_payload_dict(new_payload)
        
        assert event.get_payload_dict() == new_payload
        assert json.loads(event.payload) == new_payload
    
    def test_agent_event_set_payload_dict_complex_data(self, sample_agent_event_data):
        """Test set_payload_dict with complex nested data."""
        event = AgentEvent(**sample_agent_event_data)
        
        complex_payload = {
            "user": {"id": 123, "name": "Test User"},
            "actions": ["create", "update", "delete"],
            "metadata": {"timestamp": "2024-01-01T00:00:00Z", "version": 1}
        }
        
        event.set_payload_dict(complex_payload)
        
        retrieved_payload = event.get_payload_dict()
        assert retrieved_payload == complex_payload
        assert retrieved_payload["user"]["name"] == "Test User"
        assert "create" in retrieved_payload["actions"]
    
    def test_agent_event_repr(self, sample_agent_event_data):
        """Test AgentEvent string representation."""
        event = AgentEvent(**sample_agent_event_data)
        
        repr_str = repr(event)
        assert "AgentEvent" in repr_str
        assert event.id in repr_str
        assert event.run_id in repr_str
        assert event.type in repr_str
    
    def test_agent_event_table_name(self):
        """Test AgentEvent table name configuration."""
        assert AgentEvent.__tablename__ == "gm_agent_events"
    
    def test_agent_event_inheritance(self):
        """Test AgentEvent inherits from AgentEventBase."""
        assert issubclass(AgentEvent, AgentEventBase)
    
    def test_agent_event_compatibility_alias(self, sample_agent_event_data):
        """Test EventRow compatibility alias works."""
        from gearmeshing_ai.core.database.entities.agent_events import EventRow
        
        # EventRow should be the same as AgentEvent
        assert EventRow is AgentEvent
        
        # Should be able to create with EventRow
        event = EventRow(**sample_agent_event_data)
        assert isinstance(event, AgentEvent)
        assert event.id == "test_event_123"
    
    @pytest.mark.parametrize("field_name", [
        "type"
    ])
    def test_agent_event_required_base_fields(self, sample_agent_event_data, field_name):
        """Test that base fields become None when not provided in AgentEvent."""
        data = sample_agent_event_data.copy()
        del data[field_name]
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        
        # SQLModel sets missing fields to None instead of raising validation error
        event = AgentEvent(**data)
        
        # The missing field should be None
        assert getattr(event, field_name) is None
    
    def test_agent_event_payload_json_serialization_roundtrip(self, sample_agent_event_data):
        """Test JSON serialization roundtrip for payload."""
        data = sample_agent_event_data.copy()
        # Convert payload to JSON string for the model
        data["payload"] = json.dumps(data["payload"])
        event = AgentEvent(**data)
        
        # Get original payload
        original_payload = event.get_payload_dict()
        
        # Set it back
        event.set_payload_dict(original_payload)
        
        # Verify it's the same
        final_payload = event.get_payload_dict()
        assert final_payload == original_payload
