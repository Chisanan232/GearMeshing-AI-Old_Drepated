"""Unit tests for approval entity model.

Tests SQLModel validation, field constraints, and business logic
for the Approval entity.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.approvals import Approval


class TestApproval:
    """Tests for Approval entity model."""

    def test_approval_creation_valid_data(self):
        """Test Approval creation with valid data."""
        data = {
            "id": "approval_123",
            "run_id": "run_456",
            "risk": "high",
            "capability": "file.write",
            "reason": "Attempting to write to system file",
            "requested_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=24),
            "decision": "approved",
            "decided_at": datetime.utcnow(),
            "decided_by": "admin_user",
        }

        approval = Approval(**data)

        assert approval.id == "approval_123"
        assert approval.run_id == "run_456"
        assert approval.risk == "high"
        assert approval.capability == "file.write"
        assert approval.decision == "approved"
        assert approval.decided_by == "admin_user"

    def test_approval_creation_minimal_data(self):
        """Test Approval creation with minimal required data."""
        data = {
            "id": "approval_minimal",
            "run_id": "run_456",
            "risk": "medium",
            "capability": "api.call",
            "reason": "Making external API call",
            "requested_at": datetime.utcnow(),
        }

        approval = Approval(**data)

        assert approval.id == "approval_minimal"
        assert approval.run_id == "run_456"
        assert approval.risk == "medium"
        assert approval.capability == "api.call"
        assert approval.decision is None  # Optional field
        assert approval.decided_at is None  # Optional field
        assert approval.decided_by is None  # Optional field
        assert approval.expires_at is None  # Optional field

    def test_approval_missing_required_fields(self):
        """Test Approval validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Approval()

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}

        expected_fields = {"id", "run_id", "risk", "capability", "reason", "requested_at"}
        assert expected_fields.issubset(error_fields)

    def test_approval_string_field_length_validation(self):
        """Test Approval string field length constraints."""
        # Test ID field length
        data = {
            "id": "x" * 65,  # Exceeds max_length=64
            "run_id": "run_456",
            "risk": "medium",
            "capability": "api.call",
            "reason": "Test reason",
            "requested_at": datetime.utcnow(),
        }

        with pytest.raises(ValidationError):
            Approval(**data)

        # Test run_id field length
        data = {
            "id": "approval_123",
            "run_id": "x" * 65,  # Exceeds max_length=64
            "risk": "medium",
            "capability": "api.call",
            "reason": "Test reason",
            "requested_at": datetime.utcnow(),
        }

        with pytest.raises(ValidationError):
            Approval(**data)

    def test_approval_risk_levels(self):
        """Test Approval risk level validation."""
        valid_risks = ["low", "medium", "high", "critical"]

        for risk in valid_risks:
            data = {
                "id": f"approval_{risk}",
                "run_id": "run_456",
                "risk": risk,
                "capability": "test.capability",
                "reason": f"Test reason for {risk} risk",
                "requested_at": datetime.utcnow(),
            }

            approval = Approval(**data)
            assert approval.risk == risk

    def test_approval_decision_values(self):
        """Test Approval decision field validation."""
        valid_decisions = ["approved", "rejected", None]

        for decision in valid_decisions:
            data = {
                "id": f"approval_{decision or 'pending'}",
                "run_id": "run_456",
                "risk": "medium",
                "capability": "test.capability",
                "reason": "Test reason",
                "requested_at": datetime.utcnow(),
                "decision": decision,
            }

            approval = Approval(**data)
            assert approval.decision == decision

    def test_approval_timestamp_logic(self):
        """Test Approval timestamp relationships."""
        requested_time = datetime.utcnow()
        decided_time = requested_time + timedelta(hours=1)
        expires_time = requested_time + timedelta(hours=24)

        data = {
            "id": "approval_timestamps",
            "run_id": "run_456",
            "risk": "medium",
            "capability": "test.capability",
            "reason": "Test reason",
            "requested_at": requested_time,
            "expires_at": expires_time,
            "decision": "approved",
            "decided_at": decided_time,
            "decided_by": "admin_user",
        }

        approval = Approval(**data)

        assert approval.requested_at == requested_time
        assert approval.expires_at == expires_time
        assert approval.decided_at == decided_time
        assert approval.decided_at > approval.requested_at
        assert approval.expires_at > approval.requested_at

    def test_approval_expiration_logic(self):
        """Test Approval expiration scenarios."""
        # Non-expiring approval
        data = {
            "id": "approval_no_expire",
            "run_id": "run_456",
            "risk": "low",
            "capability": "safe.operation",
            "reason": "Safe operation",
            "requested_at": datetime.utcnow(),
            # expires_at is None (no expiration)
        }

        approval = Approval(**data)
        assert approval.expires_at is None

        # Expiring approval
        expires_at = datetime.utcnow() + timedelta(hours=12)
        data["id"] = "approval_expires"
        data["expires_at"] = expires_at

        expiring_approval = Approval(**data)
        assert expiring_approval.expires_at == expires_at

    def test_approval_workflow_states(self):
        """Test Approval workflow states."""
        # Pending approval
        pending_data = {
            "id": "approval_pending",
            "run_id": "run_456",
            "risk": "high",
            "capability": "dangerous.operation",
            "reason": "Needs approval",
            "requested_at": datetime.utcnow(),
        }

        pending_approval = Approval(**pending_data)
        assert pending_approval.decision is None
        assert pending_approval.decided_at is None
        assert pending_approval.decided_by is None

        # Approved approval
        approved_data = pending_data.copy()
        approved_data["id"] = "approval_approved"
        approved_data["decision"] = "approved"
        approved_data["decided_at"] = datetime.utcnow()
        approved_data["decided_by"] = "supervisor"

        approved_approval = Approval(**approved_data)
        assert approved_approval.decision == "approved"
        assert approved_approval.decided_at is not None
        assert approved_approval.decided_by == "supervisor"

        # Rejected approval
        rejected_data = pending_data.copy()
        rejected_data["id"] = "approval_rejected"
        rejected_data["decision"] = "rejected"
        rejected_data["decided_at"] = datetime.utcnow()
        rejected_data["decided_by"] = "security_team"

        rejected_approval = Approval(**rejected_data)
        assert rejected_approval.decision == "rejected"
        assert rejected_approval.decided_at is not None
        assert rejected_approval.decided_by == "security_team"

    def test_approval_capability_types(self):
        """Test different capability types."""
        capabilities = [
            "file.read",
            "file.write",
            "network.connect",
            "system.execute",
            "database.query",
            "api.call",
            "user.access",
        ]

        for capability in capabilities:
            data = {
                "id": f"approval_{capability.replace('.', '_')}",
                "run_id": "run_456",
                "risk": "medium",
                "capability": capability,
                "reason": f"Requesting {capability} access",
                "requested_at": datetime.utcnow(),
            }

            approval = Approval(**data)
            assert approval.capability == capability

    def test_approval_repr(self):
        """Test Approval string representation."""
        data = {
            "id": "approval_repr",
            "run_id": "run_456",
            "risk": "high",
            "capability": "file.write",
            "reason": "Test approval",
            "requested_at": datetime.utcnow(),
            "decision": "approved",
        }

        approval = Approval(**data)

        repr_str = repr(approval)
        assert "Approval" in repr_str
        assert approval.id in repr_str
        assert approval.risk in repr_str
        assert approval.decision in repr_str

    def test_approval_table_name(self):
        """Test Approval table name configuration."""
        assert Approval.__tablename__ == "gm_approvals"

    def test_approval_business_scenarios(self):
        """Test various business scenarios for approvals."""
        # High-risk operation requiring approval
        high_risk_data = {
            "id": "high_risk_approval",
            "run_id": "run_456",
            "risk": "critical",
            "capability": "system.delete",
            "reason": "Attempting to delete system files",
            "requested_at": datetime.utcnow(),
        }

        high_risk_approval = Approval(**high_risk_data)
        assert high_risk_approval.risk == "critical"
        assert high_risk_approval.capability == "system.delete"

        # Low-risk operation auto-approved
        low_risk_data = {
            "id": "low_risk_approval",
            "run_id": "run_456",
            "risk": "low",
            "capability": "file.read",
            "reason": "Reading configuration file",
            "requested_at": datetime.utcnow(),
            "decision": "approved",
            "decided_at": datetime.utcnow(),
            "decided_by": "system",
        }

        low_risk_approval = Approval(**low_risk_data)
        assert low_risk_approval.risk == "low"
        assert low_risk_approval.decision == "approved"
        assert low_risk_approval.decided_by == "system"

    def test_approval_run_relationship(self):
        """Test Approval to run relationship."""
        run_ids = ["run_001", "run_002", "run_003"]

        for run_id in run_ids:
            data = {
                "id": f"approval_{run_id}",
                "run_id": run_id,
                "risk": "medium",
                "capability": "test.operation",
                "reason": f"Test operation for {run_id}",
                "requested_at": datetime.utcnow(),
            }

            approval = Approval(**data)
            assert approval.run_id == run_id

    def test_approval_reason_field_validation(self):
        """Test Approval reason field with various content."""
        reasons = [
            "Simple reason",
            "This is a more detailed reason explaining why the approval is needed",
            "Complex reason with multiple parts: \n1. Security concern\n2. Resource usage\n3. Compliance requirement",
            "Reason with special characters: API key access, file system permissions, network ports",
        ]

        for reason in reasons:
            data = {
                "id": f"approval_reason_{len(reason)}",
                "run_id": "run_456",
                "risk": "medium",
                "capability": "test.capability",
                "reason": reason,
                "requested_at": datetime.utcnow(),
            }

            approval = Approval(**data)
            assert approval.reason == reason
