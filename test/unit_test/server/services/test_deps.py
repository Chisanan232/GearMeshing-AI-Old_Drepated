"""Unit tests for server services dependencies.

Tests verify that the OrchestratorDep dependency injection works correctly
with FastAPI's Depends mechanism.
"""

from unittest.mock import MagicMock, patch

from gearmeshing_ai.server.services.deps import OrchestratorDep
from gearmeshing_ai.server.services.orchestrator import (
    OrchestratorService,
    get_orchestrator,
)


class TestOrchestratorDep:
    """Test OrchestratorDep dependency injection."""

    def test_orchestrator_dep_is_annotated(self):
        """Test OrchestratorDep is an Annotated type."""
        # OrchestratorDep should be Annotated[OrchestratorService, Depends(...)]
        assert hasattr(OrchestratorDep, "__metadata__")

    def test_orchestrator_dep_has_depends(self):
        """Test OrchestratorDep includes Depends."""
        # Check that the metadata includes a Depends object
        metadata = OrchestratorDep.__metadata__
        assert len(metadata) > 0
        # Should have a Depends with get_orchestrator
        depends_obj = metadata[0]
        assert hasattr(depends_obj, "dependency")

    def test_orchestrator_dep_uses_get_orchestrator(self):
        """Test OrchestratorDep uses get_orchestrator function."""
        metadata = OrchestratorDep.__metadata__
        depends_obj = metadata[0]
        # The dependency should be get_orchestrator
        assert depends_obj.dependency == get_orchestrator

    def test_orchestrator_dep_type_annotation(self):
        """Test OrchestratorDep has correct type annotation."""
        # OrchestratorDep should be an Annotated type
        # Check that it has __metadata__ which is the signature of Annotated types
        assert hasattr(OrchestratorDep, "__metadata__")
        # Check that the first metadata item is a Depends
        metadata = OrchestratorDep.__metadata__
        assert len(metadata) > 0
        depends_obj = metadata[0]
        assert hasattr(depends_obj, "dependency")


class TestOrchestratorDepIntegration:
    """Test OrchestratorDep integration with FastAPI."""

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_can_be_used_in_fastapi_endpoint(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test OrchestratorDep can be used as a FastAPI dependency."""
        from fastapi import FastAPI

        app = FastAPI()

        # Define an endpoint using OrchestratorDep
        @app.get("/test")
        async def test_endpoint(orchestrator: OrchestratorDep):
            return {"orchestrator": type(orchestrator).__name__}

        # Verify the endpoint was created
        assert len(app.routes) > 0
        # Find the test endpoint
        test_route = next((r for r in app.routes if hasattr(r, "path") and r.path == "/test"), None)
        assert test_route is not None

    def test_orchestrator_dep_dependency_is_callable(self):
        """Test the dependency in OrchestratorDep is callable."""
        metadata = OrchestratorDep.__metadata__
        depends_obj = metadata[0]
        dependency = depends_obj.dependency

        # Should be callable
        assert callable(dependency)
        # Should be get_orchestrator
        assert dependency.__name__ == "get_orchestrator"

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_returns_orchestrator_service(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test the dependency returns OrchestratorService instance."""
        # Reset singleton
        import gearmeshing_ai.server.services.orchestrator as orch_module

        orch_module._orchestrator = None

        metadata = OrchestratorDep.__metadata__  # type: ignore[attr-defined]
        depends_obj = metadata[0]
        dependency = depends_obj.dependency

        result = dependency()
        assert isinstance(result, OrchestratorService)


class TestOrchestratorDepSingleton:
    """Test that OrchestratorDep always returns the same singleton."""

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_returns_singleton_across_calls(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test OrchestratorDep returns same instance across multiple calls."""
        # Reset singleton
        import gearmeshing_ai.server.services.orchestrator as orch_module

        orch_module._orchestrator = None

        metadata = OrchestratorDep.__metadata__  # type: ignore[attr-defined]
        depends_obj = metadata[0]
        dependency = depends_obj.dependency

        instance1 = dependency()
        instance2 = dependency()
        instance3 = dependency()

        assert instance1 is instance2
        assert instance2 is instance3


class TestOrchestratorDepImport:
    """Test importing OrchestratorDep."""

    def test_orchestrator_dep_can_be_imported(self):
        """Test OrchestratorDep can be imported from deps module."""
        from gearmeshing_ai.server.services.deps import OrchestratorDep as ImportedDep

        assert ImportedDep is not None

    def test_orchestrator_dep_import_is_same_as_direct(self):
        """Test imported OrchestratorDep is the same as direct import."""
        from gearmeshing_ai.server.services.deps import OrchestratorDep as ImportedDep

        assert ImportedDep is OrchestratorDep

    def test_get_orchestrator_can_be_imported(self):
        """Test get_orchestrator can be imported from orchestrator module."""
        from gearmeshing_ai.server.services.orchestrator import (
            get_orchestrator as imported_get_orch,
        )

        assert imported_get_orch is not None
        assert callable(imported_get_orch)


class TestOrchestratorDepWithMockFastAPI:
    """Test OrchestratorDep with mock FastAPI scenarios."""

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_in_multiple_endpoints(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test OrchestratorDep can be used in multiple endpoints."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/endpoint1")
        async def endpoint1(orchestrator: OrchestratorDep):
            return {"endpoint": 1}

        @app.get("/endpoint2")
        async def endpoint2(orchestrator: OrchestratorDep):
            return {"endpoint": 2}

        @app.post("/endpoint3")
        async def endpoint3(orchestrator: OrchestratorDep):
            return {"endpoint": 3}

        # All endpoints should be registered
        routes = [r for r in app.routes if hasattr(r, "path")]
        assert len(routes) >= 3

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_with_other_dependencies(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test OrchestratorDep can be combined with other dependencies."""
        from fastapi import FastAPI, Query

        app = FastAPI()

        @app.get("/combined")
        async def combined_endpoint(
            orchestrator: OrchestratorDep,
            limit: int = Query(100),
            offset: int = Query(0),
        ):
            return {"limit": limit, "offset": offset}

        # Endpoint should be registered
        routes = [r for r in app.routes if hasattr(r, "path") and r.path == "/combined"]
        assert len(routes) == 1


class TestOrchestratorDepTypeHints:
    """Test type hints and annotations of OrchestratorDep."""

    def test_orchestrator_dep_is_annotated_type(self):
        """Test OrchestratorDep is an Annotated type."""
        # Annotated types have __metadata__ attribute
        assert hasattr(OrchestratorDep, "__metadata__")
        # Metadata should be a tuple
        assert isinstance(OrchestratorDep.__metadata__, tuple)

    def test_orchestrator_dep_has_service_type(self):
        """Test OrchestratorDep references OrchestratorService."""
        # Get the string representation to verify it contains OrchestratorService
        type_str = str(OrchestratorDep)
        assert "OrchestratorService" in type_str

    def test_orchestrator_dep_metadata_structure(self):
        """Test OrchestratorDep metadata has correct structure."""
        metadata = OrchestratorDep.__metadata__
        assert isinstance(metadata, tuple)
        assert len(metadata) > 0

        # First metadata item should be Depends
        depends_obj = metadata[0]
        assert hasattr(depends_obj, "dependency")
        assert callable(depends_obj.dependency)


class TestOrchestratorDepEdgeCases:
    """Test edge cases and special scenarios."""

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_with_none_orchestrator(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test behavior when orchestrator is None before first call."""
        import gearmeshing_ai.server.services.orchestrator as orch_module

        # Reset to None
        orch_module._orchestrator = None
        assert orch_module._orchestrator is None

        # Call get_orchestrator
        orch = get_orchestrator()

        # Should no longer be None
        assert orch_module._orchestrator is not None
        assert orch is orch_module._orchestrator

    @patch("gearmeshing_ai.server.services.orchestrator.AgentService")
    @patch("gearmeshing_ai.server.services.orchestrator.DatabasePolicyProvider")
    @patch("gearmeshing_ai.server.services.orchestrator.StructuredPlanner")
    @patch("gearmeshing_ai.server.services.orchestrator.AgentServiceDeps")
    @patch("gearmeshing_ai.server.services.orchestrator.build_sql_repos_from_session")
    @patch("gearmeshing_ai.server.services.orchestrator.AsyncPostgresSaver")
    @patch("gearmeshing_ai.server.services.orchestrator.checkpointer_pool")
    @patch("gearmeshing_ai.server.services.orchestrator.build_default_registry")
    def test_orchestrator_dep_preserves_singleton_after_reset(
        self,
        mock_registry: MagicMock,
        mock_pool: MagicMock,
        mock_saver: MagicMock,
        mock_repos: MagicMock,
        mock_deps: MagicMock,
        mock_planner: MagicMock,
        mock_policy: MagicMock,
        mock_service: MagicMock,
    ) -> None:
        """Test singleton is preserved across multiple accesses."""
        import gearmeshing_ai.server.services.orchestrator as orch_module

        orch_module._orchestrator = None

        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        orch3 = get_orchestrator()

        assert orch1 is orch2 is orch3
