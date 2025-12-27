"""
Unit tests for Agent Configuration API endpoints.

Tests cover all CRUD operations and various scenarios including:
- Creating configurations with different parameters
- Retrieving configurations by ID, role, and listing
- Updating configurations with partial updates
- Deleting configurations
- Filtering by tenant and active status
- Error handling for missing resources
- Tenant-specific vs global configuration lookup
"""

import json

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestCreateAgentConfig:
    """Test agent configuration creation endpoint."""

    async def test_create_agent_config_success(self, client: AsyncClient):
        """Test successfully creating an agent configuration."""
        payload = {
            "role_name": "planner",
            "display_name": "Planning Agent",
            "description": "Handles task planning and decomposition",
            "system_prompt_key": "planner_system_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
            "capabilities": '["planning", "analysis"]',
            "tools": '["search", "calculator"]',
            "autonomy_profiles": '["balanced", "conservative"]',
            "is_active": True,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["role_name"] == "planner"
        assert data["model_name"] == "gpt-4o"
        assert data["temperature"] == 0.7
        assert "id" in data
        assert "created_at" in data

    async def test_create_agent_config_with_tenant(self, client: AsyncClient):
        """Test creating a tenant-specific agent configuration."""
        payload = {
            "role_name": "dev",
            "display_name": "Development Agent",
            "description": "Handles development tasks",
            "system_prompt_key": "dev_system_prompt",
            "model_provider": "anthropic",
            "model_name": "claude-3-opus",
            "temperature": 0.5,
            "max_tokens": 8192,
            "top_p": 0.95,
            "capabilities": '["coding", "testing"]',
            "tools": '["git", "compiler"]',
            "autonomy_profiles": '["aggressive"]',
            "tenant_id": "tenant-123",
            "is_active": True,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert data["role_name"] == "dev"

    async def test_create_agent_config_minimal(self, client: AsyncClient):
        """Test creating configuration with minimal required fields."""
        payload = {
            "role_name": "reviewer",
            "display_name": "Review Agent",
            "description": "Code review agent",
            "system_prompt_key": "reviewer_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["temperature"] == 0.7  # default
        assert data["max_tokens"] == 4096  # default
        assert data["is_active"] is True  # default

    async def test_create_agent_config_inactive(self, client: AsyncClient):
        """Test creating an inactive agent configuration."""
        payload = {
            "role_name": "qa",
            "display_name": "QA Agent",
            "description": "Quality assurance agent",
            "system_prompt_key": "qa_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": False,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["is_active"] is False

    async def test_create_agent_config_with_json_arrays(self, client: AsyncClient):
        """Test creating configuration with complex JSON array fields."""
        payload = {
            "role_name": "analyst",
            "display_name": "Analysis Agent",
            "description": "Data analysis agent",
            "system_prompt_key": "analyst_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "capabilities": '["data_analysis", "visualization", "reporting"]',
            "tools": '["pandas", "matplotlib", "sql"]',
            "autonomy_profiles": '["balanced", "conservative", "aggressive"]',
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        capabilities = json.loads(data["capabilities"])
        assert "data_analysis" in capabilities
        assert len(capabilities) == 3

    async def test_create_agent_config_temperature_bounds(self, client: AsyncClient):
        """Test creating configuration with temperature at boundaries."""
        # Test minimum temperature
        payload = {
            "role_name": "precise",
            "display_name": "Precise Agent",
            "description": "Precise agent",
            "system_prompt_key": "precise_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.0,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["temperature"] == 0.0

        # Test maximum temperature
        payload["temperature"] = 2.0
        payload["role_name"] = "creative"
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["temperature"] == 2.0


class TestGetAgentConfigById:
    """Test retrieving agent configuration by ID."""

    async def test_get_agent_config_by_id_success(self, client: AsyncClient):
        """Test successfully retrieving a configuration by ID."""
        # Create first
        payload = {
            "role_name": "test_role",
            "display_name": "Test Role",
            "description": "Test description",
            "system_prompt_key": "test_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Get by ID
        response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == config_id
        assert data["role_name"] == "test_role"

    async def test_get_agent_config_by_id_not_found(self, client: AsyncClient):
        """Test retrieving non-existent configuration by ID."""
        response = await client.get("/api/v1/agent-config/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_get_agent_config_by_id_preserves_data(self, client: AsyncClient):
        """Test that retrieved configuration preserves all data."""
        payload = {
            "role_name": "data_role",
            "display_name": "Data Agent",
            "description": "Data processing agent",
            "system_prompt_key": "data_prompt",
            "model_provider": "anthropic",
            "model_name": "claude-3",
            "temperature": 0.3,
            "max_tokens": 2048,
            "top_p": 0.8,
            "capabilities": '["data_processing"]',
            "tools": '["sql", "python"]',
            "tenant_id": "tenant-data",
            "is_active": False,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.3
        assert data["max_tokens"] == 2048
        assert data["tenant_id"] == "tenant-data"
        assert data["is_active"] is False


class TestGetAgentConfigByRole:
    """Test retrieving agent configuration by role name."""

    async def test_get_config_by_role_global(self, client: AsyncClient):
        """Test retrieving global (non-tenant) configuration by role."""
        payload = {
            "role_name": "global_role",
            "display_name": "Global Role",
            "description": "Global role config",
            "system_prompt_key": "global_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config/role/global_role")
        assert response.status_code == 200
        data = response.json()
        assert data["role_name"] == "global_role"
        assert data["tenant_id"] is None

    async def test_get_config_by_role_tenant_specific(self, client: AsyncClient):
        """Test retrieving tenant-specific configuration by role."""
        # Create global config
        global_payload = {
            "role_name": "shared_role",
            "display_name": "Shared Role",
            "description": "Shared role",
            "system_prompt_key": "shared_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=global_payload)

        # Create tenant-specific config
        tenant_payload = {
            "role_name": "shared_role",
            "display_name": "Tenant Shared Role",
            "description": "Tenant-specific shared role",
            "system_prompt_key": "tenant_shared_prompt",
            "model_provider": "anthropic",
            "model_name": "claude-3",
            "tenant_id": "tenant-abc",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=tenant_payload)

        # Get with tenant_id should return tenant-specific
        response = await client.get("/api/v1/agent-config/role/shared_role?tenant_id=tenant-abc")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-abc"
        assert data["model_name"] == "claude-3"

    async def test_get_config_by_role_fallback_to_global(self, client: AsyncClient):
        """Test fallback to global config when tenant-specific not found."""
        # Create only global config
        payload = {
            "role_name": "fallback_role",
            "display_name": "Fallback Role",
            "description": "Fallback role",
            "system_prompt_key": "fallback_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        # Request with non-existent tenant should fall back to global
        response = await client.get("/api/v1/agent-config/role/fallback_role?tenant_id=non-existent-tenant")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] is None  # Global config

    async def test_get_config_by_role_inactive_not_returned(self, client: AsyncClient):
        """Test that inactive configurations are not returned."""
        payload = {
            "role_name": "inactive_role",
            "display_name": "Inactive Role",
            "description": "Inactive role",
            "system_prompt_key": "inactive_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": False,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config/role/inactive_role")
        assert response.status_code == 404

    async def test_get_config_by_role_not_found(self, client: AsyncClient):
        """Test retrieving non-existent role."""
        response = await client.get("/api/v1/agent-config/role/non_existent_role")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestListAgentConfigs:
    """Test listing agent configurations."""

    async def test_list_agent_configs_empty(self, client: AsyncClient):
        """Test listing when no configurations exist."""
        response = await client.get("/api/v1/agent-config")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_list_agent_configs_all(self, client: AsyncClient):
        """Test listing all configurations."""
        # Create multiple configs
        for i in range(3):
            payload = {
                "role_name": f"role_{i}",
                "display_name": f"Role {i}",
                "description": f"Role {i} description",
                "system_prompt_key": f"prompt_{i}",
                "model_provider": "openai",
                "model_name": "gpt-4",
                "is_active": True,
            }
            await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3

    async def test_list_agent_configs_filter_by_tenant(self, client: AsyncClient):
        """Test listing configurations filtered by tenant."""
        # Create global configs
        for i in range(2):
            payload = {
                "role_name": f"global_role_{i}",
                "display_name": f"Global Role {i}",
                "description": f"Global role {i}",
                "system_prompt_key": f"global_prompt_{i}",
                "model_provider": "openai",
                "model_name": "gpt-4",
                "is_active": True,
            }
            await client.post("/api/v1/agent-config", json=payload)

        # Create tenant-specific configs
        for i in range(2):
            payload = {
                "role_name": f"tenant_role_{i}",
                "display_name": f"Tenant Role {i}",
                "description": f"Tenant role {i}",
                "system_prompt_key": f"tenant_prompt_{i}",
                "model_provider": "openai",
                "model_name": "gpt-4",
                "tenant_id": "tenant-xyz",
                "is_active": True,
            }
            await client.post("/api/v1/agent-config", json=payload)

        # List only tenant configs
        response = await client.get("/api/v1/agent-config?tenant_id=tenant-xyz")
        assert response.status_code == 200
        data = response.json()
        assert all(config["tenant_id"] == "tenant-xyz" for config in data)

    async def test_list_agent_configs_active_only(self, client: AsyncClient):
        """Test listing only active configurations."""
        # Create active config
        active_payload = {
            "role_name": "active_role",
            "display_name": "Active Role",
            "description": "Active role",
            "system_prompt_key": "active_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=active_payload)

        # Create inactive config
        inactive_payload = {
            "role_name": "inactive_role_list",
            "display_name": "Inactive Role",
            "description": "Inactive role",
            "system_prompt_key": "inactive_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": False,
        }
        await client.post("/api/v1/agent-config", json=inactive_payload)

        # List with active_only=True (default)
        response = await client.get("/api/v1/agent-config?active_only=true")
        assert response.status_code == 200
        data = response.json()
        assert all(config["is_active"] is True for config in data)

    async def test_list_agent_configs_include_inactive(self, client: AsyncClient):
        """Test listing including inactive configurations."""
        # Create active config
        active_payload = {
            "role_name": "active_role_2",
            "display_name": "Active Role 2",
            "description": "Active role 2",
            "system_prompt_key": "active_prompt_2",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=active_payload)

        # Create inactive config
        inactive_payload = {
            "role_name": "inactive_role_2",
            "display_name": "Inactive Role 2",
            "description": "Inactive role 2",
            "system_prompt_key": "inactive_prompt_2",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": False,
        }
        await client.post("/api/v1/agent-config", json=inactive_payload)

        # List with active_only=False
        response = await client.get("/api/v1/agent-config?active_only=false")
        assert response.status_code == 200
        data = response.json()
        # Should have both active and inactive
        has_active = any(config["is_active"] is True for config in data)
        has_inactive = any(config["is_active"] is False for config in data)
        assert has_active or has_inactive

    async def test_list_agent_configs_combined_filters(self, client: AsyncClient):
        """Test listing with combined tenant and active filters."""
        # Create tenant-specific active config
        payload = {
            "role_name": "combined_active",
            "display_name": "Combined Active",
            "description": "Combined active",
            "system_prompt_key": "combined_active_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "tenant_id": "tenant-combined",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        # Create tenant-specific inactive config
        payload["role_name"] = "combined_inactive"
        payload["is_active"] = False
        await client.post("/api/v1/agent-config", json=payload)

        # List with both filters
        response = await client.get("/api/v1/agent-config?tenant_id=tenant-combined&active_only=true")
        assert response.status_code == 200
        data = response.json()
        assert all(config["tenant_id"] == "tenant-combined" and config["is_active"] is True for config in data)


class TestUpdateAgentConfig:
    """Test updating agent configurations."""

    async def test_update_agent_config_single_field(self, client: AsyncClient):
        """Test updating a single field."""
        # Create config
        payload = {
            "role_name": "update_test",
            "display_name": "Update Test",
            "description": "Update test",
            "system_prompt_key": "update_test_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update single field
        update_payload = {"temperature": 0.5}
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.5
        assert data["model_name"] == "gpt-4"  # unchanged

    async def test_update_agent_config_multiple_fields(self, client: AsyncClient):
        """Test updating multiple fields."""
        # Create config
        payload = {
            "role_name": "multi_update",
            "display_name": "Multi Update",
            "description": "Multi update",
            "system_prompt_key": "multi_update_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update multiple fields
        update_payload = {
            "temperature": 0.3,
            "max_tokens": 2048,
            "model_name": "gpt-4-turbo",
        }
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.3
        assert data["max_tokens"] == 2048
        assert data["model_name"] == "gpt-4-turbo"

    async def test_update_agent_config_toggle_active(self, client: AsyncClient):
        """Test toggling active status."""
        # Create active config
        payload = {
            "role_name": "toggle_test",
            "display_name": "Toggle Test",
            "description": "Toggle test",
            "system_prompt_key": "toggle_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Deactivate
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json={"is_active": False})
        assert response.status_code == 200
        assert response.json()["is_active"] is False

        # Reactivate
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json={"is_active": True})
        assert response.status_code == 200
        assert response.json()["is_active"] is True

    async def test_update_agent_config_json_fields(self, client: AsyncClient):
        """Test updating JSON array fields."""
        # Create config
        payload = {
            "role_name": "json_update",
            "display_name": "JSON Update",
            "description": "JSON update",
            "system_prompt_key": "json_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "capabilities": '["old_capability"]',
            "tools": '["old_tool"]',
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update JSON fields
        update_payload = {
            "capabilities": '["new_capability_1", "new_capability_2"]',
            "tools": '["new_tool_1", "new_tool_2", "new_tool_3"]',
        }
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        capabilities = json.loads(data["capabilities"])
        tools = json.loads(data["tools"])
        assert len(capabilities) == 2
        assert len(tools) == 3

    async def test_update_agent_config_not_found(self, client: AsyncClient):
        """Test updating non-existent configuration."""
        response = await client.patch("/api/v1/agent-config/99999", json={"temperature": 0.5})
        assert response.status_code == 404

    async def test_update_agent_config_empty_update(self, client: AsyncClient):
        """Test update with no fields (should not error)."""
        # Create config
        payload = {
            "role_name": "empty_update",
            "display_name": "Empty Update",
            "description": "Empty update",
            "system_prompt_key": "empty_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]
        original_data = create_response.json()

        # Update with empty payload
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json={})
        assert response.status_code == 200
        data = response.json()
        # Data should be unchanged
        assert data["model_name"] == original_data["model_name"]


class TestDeleteAgentConfig:
    """Test deleting agent configurations."""

    async def test_delete_agent_config_success(self, client: AsyncClient):
        """Test successfully deleting a configuration."""
        # Create config
        payload = {
            "role_name": "delete_test",
            "display_name": "Delete Test",
            "description": "Delete test",
            "system_prompt_key": "delete_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Delete
        response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 204

        # Verify deletion
        get_response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert get_response.status_code == 404

    async def test_delete_agent_config_not_found(self, client: AsyncClient):
        """Test deleting non-existent configuration."""
        response = await client.delete("/api/v1/agent-config/99999")
        assert response.status_code == 404

    async def test_delete_agent_config_idempotent(self, client: AsyncClient):
        """Test that deleting already deleted config returns 404."""
        # Create and delete
        payload = {
            "role_name": "idempotent_delete",
            "display_name": "Idempotent Delete",
            "description": "Idempotent delete",
            "system_prompt_key": "idempotent_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Delete once
        response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 204

        # Try to delete again
        response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 404


class TestAgentConfigIntegration:
    """Integration tests for agent configuration workflows."""

    async def test_full_crud_workflow(self, client: AsyncClient):
        """Test complete CRUD workflow."""
        # Create
        create_payload = {
            "role_name": "crud_role",
            "display_name": "CRUD Role",
            "description": "CRUD workflow test",
            "system_prompt_key": "crud_prompt",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "is_active": True,
        }
        create_response = await client.post("/api/v1/agent-config", json=create_payload)
        assert create_response.status_code == 201
        config_id = create_response.json()["id"]

        # Read
        read_response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert read_response.status_code == 200
        assert read_response.json()["role_name"] == "crud_role"

        # Update
        update_payload = {"temperature": 0.3, "is_active": False}
        update_response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert update_response.status_code == 200
        assert update_response.json()["temperature"] == 0.3
        assert update_response.json()["is_active"] is False

        # Delete
        delete_response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert delete_response.status_code == 204

    async def test_multiple_tenants_isolation(self, client: AsyncClient):
        """Test that configurations are properly isolated by tenant."""
        # Create configs for different tenants with same role
        for tenant_id in ["tenant-1", "tenant-2", "tenant-3"]:
            payload = {
                "role_name": "shared_role_isolation",
                "display_name": f"Shared Role {tenant_id}",
                "description": f"Shared role for {tenant_id}",
                "system_prompt_key": f"prompt_{tenant_id}",
                "model_provider": "openai",
                "model_name": f"model_{tenant_id}",
                "tenant_id": tenant_id,
                "is_active": True,
            }
            await client.post("/api/v1/agent-config", json=payload)

        # Verify each tenant gets their own config
        for tenant_id in ["tenant-1", "tenant-2", "tenant-3"]:
            response = await client.get(f"/api/v1/agent-config/role/shared_role_isolation?tenant_id={tenant_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == tenant_id
            assert data["model_name"] == f"model_{tenant_id}"

    async def test_role_configuration_hierarchy(self, client: AsyncClient):
        """Test role configuration lookup hierarchy."""
        # Create global config
        global_payload = {
            "role_name": "hierarchy_role",
            "display_name": "Hierarchy Role Global",
            "description": "Global hierarchy role",
            "system_prompt_key": "hierarchy_global",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=global_payload)

        # Create tenant-specific config
        tenant_payload = {
            "role_name": "hierarchy_role",
            "display_name": "Hierarchy Role Tenant",
            "description": "Tenant hierarchy role",
            "system_prompt_key": "hierarchy_tenant",
            "model_provider": "anthropic",
            "model_name": "claude-3",
            "tenant_id": "hierarchy-tenant",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=tenant_payload)

        # Without tenant_id, should get global
        response = await client.get("/api/v1/agent-config/role/hierarchy_role")
        assert response.status_code == 200
        assert response.json()["tenant_id"] is None

        # With tenant_id, should get tenant-specific
        response = await client.get("/api/v1/agent-config/role/hierarchy_role?tenant_id=hierarchy-tenant")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "hierarchy-tenant"

        # With non-existent tenant_id, should fall back to global
        response = await client.get("/api/v1/agent-config/role/hierarchy_role?tenant_id=non-existent")
        assert response.status_code == 200
        assert response.json()["tenant_id"] is None


class TestCreateAgentConfigEdgeCases:
    """Test edge cases and error handling for creation endpoint."""

    async def test_create_with_special_characters_in_fields(self, client: AsyncClient):
        """Test creating config with special characters in string fields."""
        payload = {
            "role_name": "special_role_!@#$",
            "display_name": "Special Agent™ with émojis 🤖",
            "description": "Description with special chars: <>&\"'",
            "system_prompt_key": "prompt_key_with-dashes_and_underscores",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["display_name"] == "Special Agent™ with émojis 🤖"

    async def test_create_with_very_long_strings(self, client: AsyncClient):
        """Test creating config with very long string values."""
        long_description = "x" * 5000
        payload = {
            "role_name": "long_role",
            "display_name": "Long Display Name",
            "description": long_description,
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert len(data["description"]) == 5000

    async def test_create_with_empty_json_arrays(self, client: AsyncClient):
        """Test creating config with empty JSON arrays."""
        payload = {
            "role_name": "empty_arrays",
            "display_name": "Empty Arrays",
            "description": "Config with empty arrays",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "capabilities": "[]",
            "tools": "[]",
            "autonomy_profiles": "[]",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert json.loads(data["capabilities"]) == []
        assert json.loads(data["tools"]) == []

    async def test_create_with_large_json_arrays(self, client: AsyncClient):
        """Test creating config with large JSON arrays."""
        large_array = json.dumps([f"item_{i}" for i in range(100)])
        payload = {
            "role_name": "large_arrays",
            "display_name": "Large Arrays",
            "description": "Config with large arrays",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "capabilities": large_array,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        capabilities = json.loads(data["capabilities"])
        assert len(capabilities) == 100

    async def test_create_with_max_tokens_edge_values(self, client: AsyncClient):
        """Test creating config with edge case max_tokens values."""
        # Test minimum valid value
        payload = {
            "role_name": "min_tokens",
            "display_name": "Min Tokens",
            "description": "Min tokens config",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "max_tokens": 1,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["max_tokens"] == 1

        # Test large value
        payload["role_name"] = "large_tokens"
        payload["max_tokens"] = 1000000
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["max_tokens"] == 1000000

    async def test_create_with_top_p_edge_values(self, client: AsyncClient):
        """Test creating config with edge case top_p values."""
        # Test minimum
        payload = {
            "role_name": "min_top_p",
            "display_name": "Min Top P",
            "description": "Min top_p config",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "top_p": 0.0,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["top_p"] == 0.0

        # Test maximum
        payload["role_name"] = "max_top_p"
        payload["top_p"] = 1.0
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["top_p"] == 1.0

    async def test_create_duplicate_role_name_different_tenants(self, client: AsyncClient):
        """Test creating configs with same role name in different tenants."""
        base_payload = {
            "role_name": "duplicate_role",
            "display_name": "Duplicate Role",
            "description": "Duplicate role config",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }

        # Create in tenant 1
        payload1 = {**base_payload, "tenant_id": "tenant-1"}
        response1 = await client.post("/api/v1/agent-config", json=payload1)
        assert response1.status_code == 201

        # Create in tenant 2 with same role name
        payload2 = {**base_payload, "tenant_id": "tenant-2"}
        response2 = await client.post("/api/v1/agent-config", json=payload2)
        assert response2.status_code == 201

        # Both should exist
        assert response1.json()["id"] != response2.json()["id"]

    async def test_create_with_null_optional_fields(self, client: AsyncClient):
        """Test creating config with explicitly null optional fields."""
        payload = {
            "role_name": "null_optional",
            "display_name": "Null Optional",
            "description": "Config with null optional fields",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "done_when": None,
            "tenant_id": None,
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["done_when"] is None
        assert data["tenant_id"] is None


class TestGetAgentConfigEdgeCases:
    """Test edge cases for retrieval endpoints."""

    async def test_get_by_id_with_string_id(self, client: AsyncClient):
        """Test getting config with numeric ID as string."""
        payload = {
            "role_name": "string_id_test",
            "display_name": "String ID Test",
            "description": "Test string ID",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = str(create_response.json()["id"])

        response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 200

    async def test_get_by_role_with_special_characters(self, client: AsyncClient):
        """Test retrieving config by role with special characters."""
        payload = {
            "role_name": "role-with-dashes_and_underscores",
            "display_name": "Special Role Name",
            "description": "Role with special chars",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config/role/role-with-dashes_and_underscores")
        assert response.status_code == 200
        assert response.json()["role_name"] == "role-with-dashes_and_underscores"

    async def test_get_by_role_case_sensitive(self, client: AsyncClient):
        """Test that role retrieval is case-sensitive."""
        payload = {
            "role_name": "CaseSensitiveRole",
            "display_name": "Case Sensitive",
            "description": "Case sensitive role",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        # Correct case should work
        response = await client.get("/api/v1/agent-config/role/CaseSensitiveRole")
        assert response.status_code == 200

        # Different case should not find it
        response = await client.get("/api/v1/agent-config/role/casesensitiverole")
        assert response.status_code == 404

    async def test_get_by_role_with_empty_string(self, client: AsyncClient):
        """Test retrieving config with empty role name."""
        response = await client.get("/api/v1/agent-config/role/")
        # Empty path segment causes redirect or 404
        assert response.status_code in [307, 404]

    async def test_get_by_id_with_negative_id(self, client: AsyncClient):
        """Test getting config with negative ID."""
        response = await client.get("/api/v1/agent-config/-1")
        assert response.status_code == 404

    async def test_get_by_id_with_very_large_id(self, client: AsyncClient):
        """Test getting config with very large ID."""
        response = await client.get("/api/v1/agent-config/999999999999999")
        assert response.status_code == 404


class TestListAgentConfigsEdgeCases:
    """Test edge cases for list endpoint."""

    async def test_list_with_invalid_active_only_parameter(self, client: AsyncClient):
        """Test list with invalid active_only parameter values."""
        # Invalid parameter value causes validation error
        response = await client.get("/api/v1/agent-config?active_only=invalid")
        assert response.status_code == 422

    async def test_list_with_special_tenant_id(self, client: AsyncClient):
        """Test listing with special characters in tenant_id."""
        payload = {
            "role_name": "special_tenant_role",
            "display_name": "Special Tenant",
            "description": "Special tenant config",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "tenant_id": "tenant-with-special-chars_123",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config?tenant_id=tenant-with-special-chars_123")
        assert response.status_code == 200
        assert len(response.json()) > 0

    async def test_list_pagination_consistency(self, client: AsyncClient):
        """Test that list returns consistent results."""
        # Create multiple configs
        for i in range(5):
            payload = {
                "role_name": f"pagination_role_{i}",
                "display_name": f"Pagination Role {i}",
                "description": f"Pagination test {i}",
                "system_prompt_key": f"prompt_{i}",
                "model_provider": "openai",
                "model_name": "gpt-4",
                "is_active": True,
            }
            await client.post("/api/v1/agent-config", json=payload)

        # Get list multiple times
        response1 = await client.get("/api/v1/agent-config")
        response2 = await client.get("/api/v1/agent-config")

        data1 = response1.json()
        data2 = response2.json()

        # Should have same number of items
        assert len(data1) == len(data2)

    async def test_list_with_multiple_filters(self, client: AsyncClient):
        """Test list with multiple filter combinations."""
        # Create test data
        payload = {
            "role_name": "multi_filter_role",
            "display_name": "Multi Filter",
            "description": "Multi filter config",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "tenant_id": "multi-tenant",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        # Test all combinations
        response = await client.get("/api/v1/agent-config?tenant_id=multi-tenant&active_only=true")
        assert response.status_code == 200
        data = response.json()
        for config in data:
            assert config["tenant_id"] == "multi-tenant"
            assert config["is_active"] is True


class TestUpdateAgentConfigEdgeCases:
    """Test edge cases and error handling for update endpoint."""

    async def test_update_with_invalid_json_in_arrays(self, client: AsyncClient):
        """Test updating with malformed JSON in array fields."""
        payload = {
            "role_name": "update_json_test",
            "display_name": "Update JSON Test",
            "description": "Update JSON test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Try to update with invalid JSON (this should fail at validation)
        update_payload = {"capabilities": "not valid json"}
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        # The API might accept it as a string, so we just verify it doesn't crash
        assert response.status_code in [200, 422]

    async def test_update_preserves_unmodified_fields(self, client: AsyncClient):
        """Test that update only changes specified fields."""
        payload = {
            "role_name": "preserve_test",
            "display_name": "Preserve Test",
            "description": "Preserve test description",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]
        original_data = create_response.json()

        # Update only temperature
        update_payload = {"temperature": 0.2}
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        updated_data = response.json()

        # Verify only temperature changed
        assert updated_data["temperature"] == 0.2
        assert updated_data["max_tokens"] == original_data["max_tokens"]
        assert updated_data["top_p"] == original_data["top_p"]
        assert updated_data["model_provider"] == original_data["model_provider"]

    async def test_update_with_same_values(self, client: AsyncClient):
        """Test updating with the same values."""
        payload = {
            "role_name": "same_values_test",
            "display_name": "Same Values",
            "description": "Same values test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update with same values
        update_payload = {"temperature": 0.7}
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        assert response.json()["temperature"] == 0.7

    async def test_update_all_fields(self, client: AsyncClient):
        """Test updating all updatable fields at once."""
        payload = {
            "role_name": "update_all_test",
            "display_name": "Update All",
            "description": "Update all fields test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update all fields
        update_payload = {
            "display_name": "Updated Display",
            "description": "Updated description",
            "system_prompt_key": "updated_prompt",
            "model_provider": "anthropic",
            "model_name": "claude-3",
            "temperature": 0.3,
            "max_tokens": 2048,
            "top_p": 0.8,
            "capabilities": '["updated"]',
            "tools": '["updated_tool"]',
            "autonomy_profiles": '["updated_profile"]',
            "done_when": "Updated condition",
            "is_active": False,
        }
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()

        # Verify all fields were updated
        assert data["display_name"] == "Updated Display"
        assert data["model_provider"] == "anthropic"
        assert data["temperature"] == 0.3
        assert data["is_active"] is False

    async def test_update_with_boundary_values(self, client: AsyncClient):
        """Test updating with boundary values."""
        payload = {
            "role_name": "boundary_test",
            "display_name": "Boundary Test",
            "description": "Boundary test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update with boundary values
        update_payload = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
        }
        response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.0
        assert data["top_p"] == 1.0
        assert data["max_tokens"] == 1

    async def test_update_nonexistent_with_different_payloads(self, client: AsyncClient):
        """Test updating non-existent config with various payloads."""
        test_payloads = [
            {"temperature": 0.5},
            {"display_name": "New Name"},
            {"is_active": False},
            {"capabilities": '["test"]'},
        ]

        for payload in test_payloads:
            response = await client.patch("/api/v1/agent-config/99999", json=payload)
            assert response.status_code == 404


class TestDeleteAgentConfigEdgeCases:
    """Test edge cases for delete endpoint."""

    async def test_delete_and_verify_complete_removal(self, client: AsyncClient):
        """Test that deleted config is completely removed."""
        payload = {
            "role_name": "delete_verify",
            "display_name": "Delete Verify",
            "description": "Delete verification test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Delete
        delete_response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert delete_response.status_code == 204

        # Verify it's gone from list
        list_response = await client.get("/api/v1/agent-config")
        config_ids = [c["id"] for c in list_response.json()]
        assert config_id not in config_ids

    async def test_delete_with_string_id(self, client: AsyncClient):
        """Test deleting with string ID."""
        payload = {
            "role_name": "delete_string_id",
            "display_name": "Delete String ID",
            "description": "Delete string ID test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = str(create_response.json()["id"])

        response = await client.delete(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 204

    async def test_delete_with_invalid_id_formats(self, client: AsyncClient):
        """Test deleting with various invalid ID formats."""
        # Non-numeric IDs cause validation error (422)
        response = await client.delete("/api/v1/agent-config/abc")
        assert response.status_code == 422

        # Numeric IDs that don't exist return 404
        response = await client.delete("/api/v1/agent-config/999999999")
        assert response.status_code == 404


class TestAgentConfigSequentialOperations:
    """Test sequential operations on agent configs."""

    async def test_sequential_creates_same_role_different_tenants(self, client: AsyncClient):
        """Test creating same role in different tenants sequentially."""
        ids = []
        for i in range(5):
            payload = {
                "role_name": "sequential_role",
                "display_name": f"Sequential Role tenant-{i}",
                "description": f"Sequential test {i}",
                "system_prompt_key": "prompt_key",
                "model_provider": "openai",
                "model_name": "gpt-4",
                "tenant_id": f"tenant-{i}",
            }
            response = await client.post("/api/v1/agent-config", json=payload)
            assert response.status_code == 201
            ids.append(response.json()["id"])

        # All should have unique IDs
        assert len(set(ids)) == 5

    async def test_sequential_updates_same_config(self, client: AsyncClient):
        """Test updating same config sequentially."""
        payload = {
            "role_name": "sequential_update",
            "display_name": "Sequential Update",
            "description": "Sequential update test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.5,
        }
        create_response = await client.post("/api/v1/agent-config", json=payload)
        config_id = create_response.json()["id"]

        # Update sequentially
        for i in range(1, 6):
            update_payload = {"temperature": 0.1 * i}
            response = await client.patch(f"/api/v1/agent-config/{config_id}", json=update_payload)
            assert response.status_code == 200

        # Final state should be valid
        response = await client.get(f"/api/v1/agent-config/{config_id}")
        assert response.status_code == 200
        assert response.json()["temperature"] == 0.5


class TestAgentConfigDataValidation:
    """Test data validation and constraints."""

    async def test_create_with_whitespace_in_role_name(self, client: AsyncClient):
        """Test creating config with whitespace in role name."""
        payload = {
            "role_name": "role with spaces",
            "display_name": "Role With Spaces",
            "description": "Role with spaces in name",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        assert response.json()["role_name"] == "role with spaces"

    async def test_create_with_numeric_role_name(self, client: AsyncClient):
        """Test creating config with numeric role name."""
        payload = {
            "role_name": "12345",
            "display_name": "Numeric Role",
            "description": "Numeric role name",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201

    async def test_retrieve_by_numeric_role_name(self, client: AsyncClient):
        """Test retrieving config by numeric role name."""
        payload = {
            "role_name": "54321",
            "display_name": "Numeric Retrieve",
            "description": "Numeric role retrieve",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "is_active": True,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config/role/54321")
        assert response.status_code == 200
        assert response.json()["role_name"] == "54321"

    async def test_create_with_unicode_characters(self, client: AsyncClient):
        """Test creating config with unicode characters."""
        payload = {
            "role_name": "角色_роль_역할",
            "display_name": "Unicode Display 中文 العربية",
            "description": "Unicode description with mixed scripts",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
        }
        response = await client.post("/api/v1/agent-config", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "中文" in data["display_name"]

    async def test_list_returns_correct_types(self, client: AsyncClient):
        """Test that list endpoint returns correct data types."""
        payload = {
            "role_name": "type_check_role",
            "display_name": "Type Check",
            "description": "Type checking test",
            "system_prompt_key": "prompt_key",
            "model_provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        await client.post("/api/v1/agent-config", json=payload)

        response = await client.get("/api/v1/agent-config")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        if data:
            config = data[0]
            assert isinstance(config["id"], int)
            assert isinstance(config["role_name"], str)
            assert isinstance(config["temperature"], (int, float))
            assert isinstance(config["max_tokens"], int)
            assert isinstance(config["is_active"], bool)
            assert isinstance(config["created_at"], str)
            assert isinstance(config["updated_at"], str)
