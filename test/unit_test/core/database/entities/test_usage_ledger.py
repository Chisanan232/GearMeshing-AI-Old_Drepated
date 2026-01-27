"""Unit tests for usage ledger entity model.

Tests SQLModel validation, field constraints, and business logic
for the UsageLedger entity.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest
from pydantic import ValidationError

from gearmeshing_ai.core.database.entities.usage_ledger import UsageLedger


class TestUsageLedger:
    """Tests for UsageLedger entity model."""
    
    def test_usage_ledger_creation_valid_data(self):
        """Test UsageLedger creation with valid data."""
        data = {
            "id": "usage_123",
            "run_id": "run_456",
            "tenant_id": "tenant_789",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 150,
            "completion_tokens": 300,
            "total_tokens": 450,
            "cost_usd": 0.045,
            "created_at": datetime.utcnow()
        }
        
        usage = UsageLedger(**data)
        
        assert usage.id == "usage_123"
        assert usage.run_id == "run_456"
        assert usage.tenant_id == "tenant_789"
        assert usage.provider == "openai"
        assert usage.model == "gpt-4o"
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 300
        assert usage.total_tokens == 450
        assert usage.cost_usd == 0.045
    
    def test_usage_ledger_creation_minimal_data(self):
        """Test UsageLedger creation with minimal required data."""
        data = {
            "id": "usage_minimal",
            "run_id": "run_456",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
        
        usage = UsageLedger(**data)
        
        assert usage.id == "usage_minimal"
        assert usage.tenant_id is None  # Optional field
        assert usage.cost_usd is None  # Optional field
        assert usage.created_at is not None  # Auto-generated
    
    def test_usage_ledger_missing_required_fields(self):
        """Test UsageLedger validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            UsageLedger()
        
        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        
        expected_fields = {"id", "run_id", "provider", "model", "prompt_tokens", "completion_tokens", "total_tokens"}
        assert expected_fields.issubset(error_fields)
    
    def test_usage_ledger_string_field_length_validation(self):
        """Test UsageLedger string field length constraints."""
        # Test ID field length
        data = {
            "id": "x" * 65,  # Exceeds max_length=64
            "run_id": "run_456",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
        
        with pytest.raises(ValidationError):
            UsageLedger(**data)
        
        # Test run_id field length
        data = {
            "id": "usage_123",
            "run_id": "x" * 65,  # Exceeds max_length=64
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
        
        with pytest.raises(ValidationError):
            UsageLedger(**data)
    
    def test_usage_ledger_automatic_timestamps(self):
        """Test UsageLedger automatic timestamp generation."""
        data = {
            "id": "usage_auto",
            "run_id": "run_456",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
        
        usage = UsageLedger(**data)
        
        assert usage.created_at is not None
        assert isinstance(usage.created_at, datetime)
    
    def test_usage_ledger_token_validation(self):
        """Test UsageLedger token count validation."""
        # Valid token counts
        valid_tokens = [0, 1, 100, 1000, 10000, 100000]
        
        for token_count in valid_tokens:
            data = {
                "id": f"usage_tokens_{token_count}",
                "run_id": "run_456",
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": token_count,
                "completion_tokens": token_count,
                "total_tokens": token_count * 2
            }
            
            usage = UsageLedger(**data)
            assert usage.prompt_tokens == token_count
            assert usage.completion_tokens == token_count
            assert usage.total_tokens == token_count * 2
    
    def test_usage_ledger_negative_tokens(self):
        """Test UsageLedger with negative token counts."""
        data = {
            "id": "usage_negative",
            "run_id": "run_456",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": -100,
            "completion_tokens": -200,
            "total_tokens": -300
        }
        
        # Should allow negative tokens (for corrections/refunds)
        usage = UsageLedger(**data)
        assert usage.prompt_tokens == -100
        assert usage.completion_tokens == -200
        assert usage.total_tokens == -300
    
    def test_usage_ledger_cost_validation(self):
        """Test UsageLedger cost field validation."""
        # Valid costs
        valid_costs = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        for cost in valid_costs:
            data = {
                "id": f"usage_cost_{cost}",
                "run_id": "run_456",
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "cost_usd": cost
            }
            
            usage = UsageLedger(**data)
            assert usage.cost_usd == cost
    
    def test_usage_ledger_provider_validation(self):
        """Test UsageLedger provider field validation."""
        valid_providers = ["openai", "anthropic", "google", "xai", "local", "custom"]
        
        for provider in valid_providers:
            data = {
                "id": f"usage_{provider}",
                "run_id": "run_456",
                "provider": provider,
                "model": "test_model",
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
            
            usage = UsageLedger(**data)
            assert usage.provider == provider
    
    def test_usage_ledger_model_validation(self):
        """Test UsageLedger model field validation."""
        valid_models = [
            "gpt-4o",
            "gpt-4-turbo",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "grok-beta",
            "local-llm",
            "custom-model-v1"
        ]
        
        for model in valid_models:
            data = {
                "id": f"usage_model_{model.replace('-', '_')}",
                "run_id": "run_456",
                "provider": "test_provider",
                "model": model,
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
            
            usage = UsageLedger(**data)
            assert usage.model == model
    
    def test_usage_ledger_repr(self):
        """Test UsageLedger string representation."""
        data = {
            "id": "usage_repr",
            "run_id": "run_456",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 150,
            "completion_tokens": 300,
            "total_tokens": 450
        }
        
        usage = UsageLedger(**data)
        
        repr_str = repr(usage)
        assert "UsageLedger" in repr_str
        assert usage.id in repr_str
        assert usage.run_id in repr_str
        assert str(usage.total_tokens) in repr_str
    
    def test_usage_ledger_table_name(self):
        """Test UsageLedger table name configuration."""
        assert UsageLedger.__tablename__ == "gm_usage_ledger"
    
    def test_usage_ledger_run_relationship(self):
        """Test UsageLedger to run relationship."""
        run_ids = ["run_001", "run_002", "run_003"]
        
        for run_id in run_ids:
            data = {
                "id": f"usage_{run_id}",
                "run_id": run_id,
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
            
            usage = UsageLedger(**data)
            assert usage.run_id == run_id
    
    def test_usage_ledger_tenant_isolation(self):
        """Test UsageLedger tenant isolation."""
        tenants = ["tenant_a", "tenant_b", "tenant_c"]
        
        for tenant_id in tenants:
            data = {
                "id": f"usage_{tenant_id}",
                "run_id": "run_456",
                "tenant_id": tenant_id,
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
            
            usage = UsageLedger(**data)
            assert usage.tenant_id == tenant_id
    
    def test_usage_ledger_cost_calculation_scenarios(self):
        """Test various cost calculation scenarios."""
        scenarios = [
            {
                "name": "openai_gpt4o",
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "expected_cost": 0.045  # $0.03 per 1K prompt, $0.06 per 1K completion
            },
            {
                "name": "anthropic_claude",
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "expected_cost": 0.0375  # $0.015 per 1K prompt, $0.075 per 1K completion
            },
            {
                "name": "google_gemini",
                "provider": "google",
                "model": "gemini-2.0-flash",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "expected_cost": 0.025  # $0.0125 per 1K prompt, $0.025 per 1K completion
            }
        ]
        
        for scenario in scenarios:
            data = {
                "id": f"usage_{scenario['name']}",
                "run_id": "run_456",
                "tenant_id": "tenant_789",
                "provider": scenario["provider"],
                "model": scenario["model"],
                "prompt_tokens": scenario["prompt_tokens"],
                "completion_tokens": scenario["completion_tokens"],
                "total_tokens": scenario["total_tokens"],
                "cost_usd": scenario["expected_cost"]
            }
            
            usage = UsageLedger(**data)
            assert usage.provider == scenario["provider"]
            assert usage.model == scenario["model"]
            assert usage.cost_usd == scenario["expected_cost"]
    
    def test_usage_ledger_high_volume_usage(self):
        """Test UsageLedger with high volume usage."""
        high_volume_data = {
            "id": "usage_high_volume",
            "run_id": "run_456",
            "tenant_id": "tenant_789",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 50000,  # 50K prompt tokens
            "completion_tokens": 100000,  # 100K completion tokens
            "total_tokens": 150000,  # 150K total tokens
            "cost_usd": 7.5  # $7.50 total cost
        }
        
        usage = UsageLedger(**high_volume_data)
        
        assert usage.prompt_tokens == 50000
        assert usage.completion_tokens == 100000
        assert usage.total_tokens == 150000
        assert usage.cost_usd == 7.5
    
    def test_usage_ledger_zero_usage(self):
        """Test UsageLedger with zero usage (for failed calls)."""
        zero_usage_data = {
            "id": "usage_zero",
            "run_id": "run_456",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }
        
        usage = UsageLedger(**zero_usage_data)
        
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0
    
    def test_usage_ledger_refund_scenario(self):
        """Test UsageLedger refund scenario (negative costs)."""
        refund_data = {
            "id": "usage_refund",
            "run_id": "run_456",
            "tenant_id": "tenant_789",
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": -1000,  # Refunded prompt tokens
            "completion_tokens": -500,  # Refunded completion tokens
            "total_tokens": -1500,  # Refunded total tokens
            "cost_usd": -0.045  # Refunded cost
        }
        
        usage = UsageLedger(**refund_data)
        
        assert usage.prompt_tokens == -1000
        assert usage.completion_tokens == -500
        assert usage.total_tokens == -1500
        assert usage.cost_usd == -0.045
    
    def test_usage_ledger_batch_usage_tracking(self):
        """Test UsageLedger for batch operation tracking."""
        batch_entries = []
        
        for i in range(10):
            data = {
                "id": f"usage_batch_{i}",
                "run_id": "run_batch",
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 100 * (i + 1),
                "completion_tokens": 200 * (i + 1),
                "total_tokens": 300 * (i + 1),
                "cost_usd": 0.01 * (i + 1)
            }
            
            usage = UsageLedger(**data)
            batch_entries.append(usage)
        
        # Verify batch entries
        assert len(batch_entries) == 10
        total_tokens = sum(entry.total_tokens for entry in batch_entries)
        total_cost = sum(entry.cost_usd for entry in batch_entries)
        
        assert total_tokens == 16500  # Sum of 300, 600, 900, ..., 3000
        assert total_cost == 0.55   # Sum of 0.01, 0.02, ..., 0.10
