"""Unit tests for policy entity model.

Tests SQLModel validation, field constraints, and business logic
for the Policy entity.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.policies import Policy


class TestPolicy:
    """Tests for Policy entity model."""
    
    def test_policy_creation_valid_data(self):
        """Test Policy creation with valid data."""
        config_data = {
            "risk_thresholds": {
                "low": {"max_tokens": 1000, "allowed_tools": ["read"]},
                "medium": {"max_tokens": 5000, "allowed_tools": ["read", "write"]},
                "high": {"max_tokens": 1000, "require_approval": True}
            },
            "capabilities": {
                "file_access": {"allowed_paths": ["/tmp", "/home/user"]},
                "network_access": {"allowed_domains": ["api.example.com"]},
                "system_commands": {"blocked": ["rm", "sudo"]}
            },
            "autonomy_settings": {
                "max_autonomous_steps": 10,
                "require_human_confirmation": True,
                "timeout_minutes": 30
            }
        }
        
        data = {
            "id": "policy_123",
            "tenant_id": "tenant_456",
            "config": config_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        policy = Policy(**data)
        
        assert policy.id == "policy_123"
        assert policy.tenant_id == "tenant_456"
        assert policy.config == config_data
    
    def test_policy_creation_minimal_data(self):
        """Test Policy creation with minimal required data."""
        config_data = {"default_settings": {"safe_mode": True}}
        
        data = {
            "id": "policy_minimal",
            "tenant_id": "tenant_456",
            "config": config_data
        }
        
        policy = Policy(**data)
        
        assert policy.id == "policy_minimal"
        assert policy.tenant_id == "tenant_456"
        assert policy.config == config_data
        assert policy.created_at is not None  # Auto-generated
        assert policy.updated_at is not None   # Auto-generated
    
    def test_policy_missing_required_fields(self):
        """Test Policy validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Policy()
        
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        
        expected_fields = {"id", "tenant_id", "config"}
        assert expected_fields.issubset(error_fields)
    
    def test_policy_string_field_length_validation(self):
        """Test Policy string field length constraints."""
        # Test ID field length
        data = {
            "id": "x" * 65,  # Exceeds max_length=64
            "tenant_id": "tenant_456",
            "config": {}
        }
        
        with pytest.raises(ValidationError):
            Policy(**data)
        
        # Test tenant_id field length
        data = {
            "id": "policy_123",
            "tenant_id": "x" * 129,  # Exceeds max_length=128
            "config": {}
        }
        
        with pytest.raises(ValidationError):
            Policy(**data)
    
    def test_policy_automatic_timestamps(self):
        """Test Policy automatic timestamp generation."""
        data = {
            "id": "policy_auto",
            "tenant_id": "tenant_456",
            "config": {}
        }
        
        policy = Policy(**data)
        
        assert policy.created_at is not None
        assert policy.updated_at is not None
        assert isinstance(policy.created_at, datetime)
        assert isinstance(policy.updated_at, datetime)
    
    def test_policy_complex_config_data(self):
        """Test Policy with complex configuration data."""
        complex_config = {
            "security": {
                "encryption": {
                    "at_rest": True,
                    "in_transit": True,
                    "algorithm": "AES-256"
                },
                "access_control": {
                    "authentication": {
                        "required": True,
                        "methods": ["mfa", "sso"],
                        "session_timeout": 3600
                    },
                    "authorization": {
                        "rbac_enabled": True,
                        "default_permissions": ["read"],
                        "admin_permissions": ["read", "write", "delete", "admin"]
                    }
                }
            },
            "resource_limits": {
                "compute": {
                    "max_cpu_cores": 8,
                    "max_memory_gb": 32,
                    "max_disk_gb": 100,
                    "max_concurrent_processes": 10
                },
                "network": {
                    "max_bandwidth_mbps": 1000,
                    "allowed_ports": [80, 443, 22],
                    "blocked_protocols": ["ftp", "telnet"]
                },
                "storage": {
                    "max_file_size_mb": 100,
                    "max_total_storage_gb": 1000,
                    "retention_days": 30
                }
            },
            "monitoring": {
                "logging": {
                    "level": "INFO",
                    "retention_days": 90,
                    "include_sensitive_data": False
                },
                "metrics": {
                    "collection_interval_seconds": 60,
                    "enabled_metrics": ["cpu", "memory", "disk", "network"],
                    "alert_thresholds": {
                        "cpu_usage_percent": 80,
                        "memory_usage_percent": 85,
                        "disk_usage_percent": 90
                    }
                }
            },
            "compliance": {
                "standards": ["SOC2", "GDPR", "HIPAA"],
                "data_classification": {
                    "public": {"encryption_required": False, "audit_required": False},
                    "internal": {"encryption_required": True, "audit_required": False},
                    "confidential": {"encryption_required": True, "audit_required": True},
                    "restricted": {"encryption_required": True, "audit_required": True, "access_log": True}
                },
                "audit_requirements": {
                    "log_all_access": True,
                    "log_modifications": True,
                    "log_exports": True,
                    "audit_retention_years": 7
                }
            }
        }
        
        data = {
            "id": "policy_complex",
            "tenant_id": "tenant_456",
            "config": complex_config
        }
        
        policy = Policy(**data)
        
        assert policy.config == complex_config
        assert policy.config["security"]["encryption"]["algorithm"] == "AES-256"
        assert policy.config["resource_limits"]["compute"]["max_cpu_cores"] == 8
        assert len(policy.config["compliance"]["standards"]) == 3
    
    def test_policy_config_validation_scenarios(self):
        """Test various policy configuration scenarios."""
        scenarios = [
            {
                "name": "strict_security",
                "config": {
                    "security_level": "high",
                    "require_approval_for_all": True,
                    "blocked_operations": ["file_delete", "system_modify"],
                    "audit_all_actions": True
                }
            },
            {
                "name": "permissive_development",
                "config": {
                    "security_level": "low",
                    "allow_self_approval": True,
                    "allowed_operations": ["*"],
                    "audit_critical_only": True
                }
            },
            {
                "name": "balanced_production",
                "config": {
                    "security_level": "medium",
                    "approval_threshold": "medium_risk_and_above",
                    "allowed_operations": ["read", "write", "compute"],
                    "audit_frequency": "daily"
                }
            }
        ]
        
        for scenario in scenarios:
            data = {
                "id": f"policy_{scenario['name']}",
                "tenant_id": "tenant_456",
                "config": scenario["config"]
            }
            
            policy = Policy(**data)
            assert policy.config == scenario["config"]
            assert policy.config["security_level"] == scenario["config"]["security_level"]
    
    def test_policy_tenant_isolation(self):
        """Test policy tenant isolation."""
        tenants = ["tenant_a", "tenant_b", "tenant_c"]
        
        for tenant_id in tenants:
            data = {
                "id": f"policy_{tenant_id}",
                "tenant_id": tenant_id,
                "config": {"tenant_specific": True, "settings": {"mode": "custom"}}
            }
            
            policy = Policy(**data)
            assert policy.tenant_id == tenant_id
            assert policy.config["tenant_specific"] is True
    
    def test_policy_default_config(self):
        """Test policy with default configuration."""
        default_config = {
            "version": "1.0",
            "default_behavior": "safe",
            "fallback_policy": "deny"
        }
        
        data = {
            "id": "policy_default",
            "tenant_id": "tenant_456",
            "config": default_config
        }
        
        policy = Policy(**data)
        
        assert policy.config["version"] == "1.0"
        assert policy.config["default_behavior"] == "safe"
        assert policy.config["fallback_policy"] == "deny"
    
    def test_policy_config_updates(self):
        """Test policy configuration updates."""
        initial_config = {"version": "1.0", "mode": "strict"}
        updated_config = {"version": "2.0", "mode": "balanced", "new_feature": True}
        
        # Create policy with initial config
        data = {
            "id": "policy_updatable",
            "tenant_id": "tenant_456",
            "config": initial_config,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        policy = Policy(**data)
        
        # Update config
        policy.config = updated_config
        policy.updated_at = datetime.utcnow()
        
        assert policy.config == updated_config
        assert policy.config["version"] == "2.0"
        assert policy.config["new_feature"] is True
    
    def test_policy_repr(self):
        """Test Policy string representation."""
        data = {
            "id": "policy_repr",
            "tenant_id": "tenant_456",
            "config": {"mode": "test"}
        }
        
        policy = Policy(**data)
        
        repr_str = repr(policy)
        assert "Policy" in repr_str
        assert policy.id in repr_str
        assert policy.tenant_id in repr_str
    
    def test_policy_table_name(self):
        """Test Policy table name configuration."""
        assert Policy.__tablename__ == "gm_policies"
    
    def test_policy_config_edge_cases(self):
        """Test policy configuration edge cases."""
        edge_cases = [
            {"empty_config": {}},
            {"null_values": {"setting1": None, "setting2": None}},
            {"nested_empty": {"level1": {"level2": {}}}},
            {"large_numbers": {"max_value": 9223372036854775807}},
            {"special_chars": {"path": "/tmp/test@#$%", "unicode": "测试中文"}},
            {"boolean_flags": {"flag1": True, "flag2": False, "flag3": True}}
        ]
        
        for case in edge_cases:
            data = {
                "id": f"policy_edge_{list(case.keys())[0]}",
                "tenant_id": "tenant_456",
                "config": case
            }
            
            policy = Policy(**data)
            assert policy.config == case
    
    def test_policy_business_logic_validation(self):
        """Test business logic validation for policy configurations."""
        # Valid business policy
        valid_business_config = {
            "business_hours": {
                "start": "09:00",
                "end": "17:00",
                "timezone": "UTC",
                "weekends_allowed": False
            },
            "approval_workflow": {
                "auto_approve_below_risk": "low",
                "require_manager_for": ["high", "critical"],
                "executive_approval_for": "critical"
            },
            "cost_controls": {
                "monthly_budget": 10000,
                "alert_threshold": 0.8,
                "auto_block_at_limit": True
            }
        }
        
        data = {
            "id": "policy_business",
            "tenant_id": "tenant_456",
            "config": valid_business_config
        }
        
        policy = Policy(**data)
        
        assert policy.config["business_hours"]["start"] == "09:00"
        assert policy.config["approval_workflow"]["auto_approve_below_risk"] == "low"
        assert policy.config["cost_controls"]["monthly_budget"] == 10000
