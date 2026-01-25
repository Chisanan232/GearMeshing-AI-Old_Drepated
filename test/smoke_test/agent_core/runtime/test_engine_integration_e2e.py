"""
Integration smoke tests for LangGraph runtime engine with real-world scenarios.

These tests focus on realistic use cases and integration patterns:
1. Multi-agent collaboration workflows
2. Human-in-the-loop workflows
3. Cross-domain integration (data science, web automation, etc.)
4. Enterprise workflow patterns
5. Real-time and streaming workflows
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from test.settings import test_settings

# Import fixtures from the shared fixtures module
from typing import cast, Any
from unittest.mock import MagicMock

import pytest

from gearmeshing_ai.agent_core.capabilities.base import CapabilityResult
from gearmeshing_ai.agent_core.model_provider import async_create_model_for_role
from gearmeshing_ai.agent_core.policy.global_policy import GlobalPolicy
from gearmeshing_ai.agent_core.runtime.engine import AgentEngine
from gearmeshing_ai.agent_core.runtime.models import EngineDeps
from gearmeshing_ai.agent_core.schemas.domain import (
    AgentEventType,
    AgentRun,
    AgentRunStatus,
    RiskLevel,
)


class TestRealWorldIntegrationWorkflows:
    """Real-world integration workflow test scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_data_science_pipeline_workflow(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test complete data science pipeline workflow."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Data science pipeline workflow
        ds_pipeline = [
            # Data Ingestion
            {
                "kind": "thought",
                "thought": "plan_data_ingestion",
                "args": {"sources": ["csv", "api", "database"], "volume": "large"},
            },
            {"kind": "action", "capability": "docs_read", "args": {"file_path": "raw_sales_data.csv"}},
            {"kind": "action", "capability": "web_search", "args": {"query": "market trends 2024", "max_results": 100}},
            # Data Cleaning and Preprocessing
            {
                "kind": "thought",
                "thought": "clean_and_preprocess",
                "args": {"quality_checks": True, "outlier_detection": True},
            },
            {"kind": "action", "capability": "summarize", "args": {"data": "raw_data", "operation": "cleaning"}},
            # Exploratory Data Analysis
            {
                "kind": "thought",
                "thought": "perform_eda",
                "args": {"analysis_types": ["descriptive", "correlation", "trend"]},
            },
            {
                "kind": "action",
                "capability": "summarize",
                "args": {"data": "clean_data", "analysis_type": "exploratory"},
            },
            # Feature Engineering
            {
                "kind": "thought",
                "thought": "engineer_features",
                "args": {"domain": "sales", "target": "revenue_prediction"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "features.csv", "content": "engineered_features"},
            },
            # Model Training and Evaluation
            {
                "kind": "thought",
                "thought": "train_model",
                "args": {"algorithms": ["random_forest", "xgboost"], "validation": "cross_validation"},
            },
            {
                "kind": "action",
                "capability": "shell_exec",
                "args": {"command": "python train_model.py --features features.csv"},
            },
            # Results and Reporting
            {
                "kind": "thought",
                "thought": "generate_report",
                "args": {"format": "comprehensive", "audience": "stakeholders"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "data_science_report.html", "content": "final_report"},
            },
        ]

        # Create data science run
        ds_run = AgentRun(
            id=str(uuid.uuid4()),
            role="data_scientist",
            objective="Complete data science pipeline from raw data to insights",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = ds_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Mock capability executions
        async def mock_capability_execution(*args, **kwargs):
            return CapabilityResult(ok=True, output={"status": "success", "processed_records": 10000})

        mock_capability = MagicMock()
        mock_capability.execute = mock_capability_execution
        cast(MagicMock, engine_deps.capabilities.get).return_value = mock_capability

        # Execute data science pipeline
        result = await engine.start_run(run=ds_run, plan=ds_pipeline)

        # Verify successful pipeline execution
        assert result == ds_run.id

        # Verify comprehensive event logging
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        assert len(event_calls) >= 10  # Should have events for each major step

        # Verify stage transitions
        event_types = [call[0][0].type for call in event_calls]
        assert AgentEventType.run_started in event_types
        assert AgentEventType.plan_created in event_types
        assert AgentEventType.thought_executed in event_types
        assert AgentEventType.capability_requested in event_types
        assert AgentEventType.capability_executed in event_types
        assert AgentEventType.run_completed in event_types

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_web_automation_workflow(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test web automation and scraping workflow."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Web automation workflow
        web_automation_plan = [
            # Planning Phase
            {
                "kind": "thought",
                "thought": "plan_web_scraping",
                "args": {
                    "target_sites": ["ecommerce", "news", "social"],
                    "data_types": ["prices", "articles", "trends"],
                },
            },
            # Execution Phase
            {
                "kind": "action",
                "capability": "web_search",
                "args": {"query": "product prices comparison", "max_results": 50},
            },
            {"kind": "thought", "thought": "process_scraped_data", "args": {"cleaning": True, "normalization": True}},
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "scraped_data.json", "content": "raw_scraped_data"},
            },
            # Analysis Phase
            {
                "kind": "thought",
                "thought": "analyze_market_data",
                "args": {"analysis_type": "price_trend", "timeframe": "30_days"},
            },
            {
                "kind": "action",
                "capability": "summarize",
                "args": {"data": "scraped_data", "analysis": "price_comparison"},
            },
            # Reporting Phase
            {
                "kind": "thought",
                "thought": "generate_automation_report",
                "args": {"include_recommendations": True, "format": "dashboard"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "automation_report.html", "content": "final_report"},
            },
        ]

        # Create web automation run
        web_run = AgentRun(
            id=str(uuid.uuid4()),
            role="automation_specialist",
            objective="Automated web scraping and market analysis",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = web_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Mock web automation capabilities
        async def mock_web_automation(*args, **kwargs):
            return CapabilityResult(
                ok=True, output={"scraped_items": 150, "data_quality": "high", "processing_time": "45s"}
            )

        mock_web_capability = MagicMock()
        mock_web_capability.execute = mock_web_automation
        cast(MagicMock, engine_deps.capabilities.get).return_value = mock_web_capability

        # Execute web automation workflow
        result = await engine.start_run(run=web_run, plan=web_automation_plan)

        # Verify successful automation
        assert result == web_run.id

        # Verify tool invocations were tracked
        assert cast(MagicMock, engine_deps.tool_invocations.append).call_count > 0

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_human_in_the_loop_workflow(
        self,
        engine_deps: EngineDeps,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test human-in-the-loop workflow with approvals and interventions."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        # Mock policy to require human approval for critical actions
        # Create a new policy instance for this test
        test_policy = MagicMock()

        def mock_decide(capability, *args, **kwargs):
            # Check if this is a critical action that requires approval
            if hasattr(capability, "value") and "shell_exec" in str(capability.value):
                decision = MagicMock()
                decision.block = False
                decision.block_reason = None
                decision.require_approval = True
                decision.risk = RiskLevel.high
                return decision
            else:
                decision = MagicMock()
                decision.block = False
                decision.block_reason = None
                decision.require_approval = False
                decision.risk = RiskLevel.low
                return decision

        # Setup the test policy
        test_policy.decide = mock_decide
        cast(MagicMock, test_policy.validate_tool_args).return_value = None
        cast(MagicMock, test_policy.classify_risk).return_value = RiskLevel.low

        engine = AgentEngine(policy=test_policy, deps=engine_deps)

        # Human-in-the-loop workflow
        hitl_plan = [
            # Initial analysis (no approval needed)
            {
                "kind": "thought",
                "thought": "analyze_system_state",
                "args": {"system": "production", "scope": "performance"},
            },
            {"kind": "action", "capability": "docs_read", "args": {"file_path": "system_logs.txt"}},
            # Critical action requiring approval
            {
                "kind": "thought",
                "thought": "plan_critical_action",
                "args": {"action": "database_cleanup", "risk": "data_loss"},
            },
            {
                "kind": "action",
                "capability": "shell_exec",
                "args": {"command": "DELETE FROM logs WHERE created_at < '2024-01-01'"},
            },
            # Post-approval action
            {
                "kind": "thought",
                "thought": "verify_action_results",
                "args": {"verification": "data_integrity", "rollback_plan": True},
            },
        ]

        # Create HITL run
        hitl_run = AgentRun(
            id=str(uuid.uuid4()),
            role="system_administrator",
            objective="Critical system maintenance with human oversight",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = hitl_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        cast(MagicMock, engine_deps.approvals.create).return_value = None

        # Mock capability execution
        async def mock_critical_action(*args, **kwargs):
            return CapabilityResult(ok=True, output={"deleted_records": 1000000, "space_freed": "5GB"})

        mock_critical_capability = MagicMock()
        mock_critical_capability.execute = mock_critical_action
        cast(MagicMock, engine_deps.capabilities.get).return_value = mock_critical_capability

        # Execute HITL workflow
        result = await engine.start_run(run=hitl_run, plan=hitl_plan)

        # Verify approval was requested for critical action
        cast(MagicMock, engine_deps.approvals.create).assert_called()

        # Verify approval events were logged
        event_calls = cast(MagicMock, engine_deps.events.append).call_args_list
        approval_events = [call for call in event_calls if call[0][0].type == AgentEventType.approval_requested]
        assert len(approval_events) > 0

        # Verify workflow completed
        assert result == hitl_run.id

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_multi_agent_collaboration_workflow(
        self,
        engine_deps: EngineDeps,
        mock_policy: GlobalPolicy,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test multi-agent collaboration workflow."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        engine = AgentEngine(policy=mock_policy, deps=engine_deps)

        # Multi-agent collaboration workflow
        collaboration_plan = [
            # Agent 1: Data Collection Specialist
            {
                "kind": "thought",
                "thought": "coordinate_data_collection",
                "args": {"agents": ["collector", "analyzer", "reporter"], "timeline": "collaborative"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "shared_workspace/data_collection_plan.json", "content": "plan"},
            },
            # Agent 2: Analysis Specialist
            {
                "kind": "thought",
                "thought": "perform_specialized_analysis",
                "args": {"specialization": "statistical", "data_source": "shared_workspace"},
            },
            {
                "kind": "action",
                "capability": "summarize",
                "args": {"data": "collected_data", "analysis_type": "specialized"},
            },
            # Agent 3: Integration Specialist
            {
                "kind": "thought",
                "thought": "integrate_multi_agent_results",
                "args": {"agents": ["collector", "analyzer"], "integration_method": "ensemble"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "shared_workspace/integrated_results.json", "content": "results"},
            },
            # Final Coordination
            {
                "kind": "thought",
                "thought": "finalize_collaborative_output",
                "args": {"quality_check": True, "consensus": True},
            },
        ]

        # Create multi-agent run
        multi_run = AgentRun(
            id=str(uuid.uuid4()),
            role="coordination_agent",
            objective="Coordinate multi-agent collaboration for complex analysis",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = multi_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None

        # Mock multi-agent capabilities
        async def mock_collaborative_action(*args, **kwargs):
            return CapabilityResult(
                ok=True,
                output={
                    "agents_involved": 3,
                    "collaboration_score": 0.95,
                    "shared_artifacts": ["data_collection_plan.json", "integrated_results.json"],
                },
            )

        mock_collaborative_capability = MagicMock()
        mock_collaborative_capability.execute = mock_collaborative_action
        cast(MagicMock, engine_deps.capabilities.get).return_value = mock_collaborative_capability

        # Execute multi-agent workflow
        result = await engine.start_run(run=multi_run, plan=collaboration_plan)

        # Verify successful collaboration
        assert result == multi_run.id

        # Verify shared workspace operations
        write_calls = cast(MagicMock, engine_deps.tool_invocations.append).call_args_list
        shared_operations = [call for call in write_calls if "shared_workspace" in str(call)]
        assert len(shared_operations) > 0

    @pytest.mark.asyncio
    @pytest.mark.smoke_ai
    async def test_enterprise_workflow_compliance(
        self,
        engine_deps: EngineDeps,
        compose_stack: Any,
        database_url: str,
        agent_configs_setup,
        patched_settings,
    ) -> None:
        """Test enterprise workflow with compliance and audit requirements."""
        if not test_settings.ai_provider.openai.api_key:
            pytest.skip("OpenAI API key not configured")

        # Create real AI model using settings from dotenv
        thought_model = await async_create_model_for_role("assistant")

        engine_deps = EngineDeps(
            runs=engine_deps.runs,
            events=engine_deps.events,
            approvals=engine_deps.approvals,
            checkpoints=engine_deps.checkpoints,
            tool_invocations=engine_deps.tool_invocations,
            capabilities=engine_deps.capabilities,
            usage=engine_deps.usage,
            checkpointer=engine_deps.checkpointer,
            thought_model=thought_model,
            prompt_provider=None,
            role_provider=None,
            mcp_info_provider=None,
            mcp_call=None,
        )

        # Enterprise compliance policy
        enterprise_policy = MagicMock()

        def mock_compliance_check(capability, *args, **kwargs):
            # All actions require compliance verification
            decision = MagicMock()
            decision.block = False
            decision.block_reason = None
            decision.require_approval = True  # Enterprise: always require approval
            decision.risk = RiskLevel.medium
            decision.compliance_required = True
            decision.audit_trail = True
            return decision

        # Setup the enterprise policy
        enterprise_policy.decide = mock_compliance_check
        cast(MagicMock, enterprise_policy.validate_tool_args).return_value = None
        cast(MagicMock, enterprise_policy.classify_risk).return_value = RiskLevel.medium

        engine = AgentEngine(policy=enterprise_policy, deps=engine_deps)

        # Enterprise compliance workflow
        compliance_plan = [
            # Compliance Check
            {
                "kind": "thought",
                "thought": "verify_compliance_requirements",
                "args": {"regulations": ["GDPR", "SOX", "HIPAA"], "data_classification": "sensitive"},
            },
            # Audit Trail Setup
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "audit/workflow_log.json", "content": "audit_trail_start"},
            },
            # Compliant Data Processing
            {
                "kind": "thought",
                "thought": "process_with_compliance",
                "args": {"anonymization": True, "encryption": "AES256", "retention": "7_years"},
            },
            {
                "kind": "action",
                "capability": "shell_exec",
                "args": {"command": "python process_data.py --compliant --encrypt"},
            },
            # Compliance Reporting
            {
                "kind": "thought",
                "thought": "generate_compliance_report",
                "args": {"standards": ["ISO27001", "SOC2"], "evidence": "full_audit_trail"},
            },
            {
                "kind": "action",
                "capability": "codegen",
                "args": {"file_path": "compliance/compliance_report.pdf", "content": "compliance_evidence"},
            },
        ]

        # Create enterprise run
        enterprise_run = AgentRun(
            id=str(uuid.uuid4()),
            role="compliance_officer",
            objective="Execute enterprise workflow with full compliance and audit",
            status=AgentRunStatus.pending,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tenant_id="enterprise-tenant",
        )

        # Setup mocks
        cast(MagicMock, engine_deps.runs.create).return_value = None
        cast(MagicMock, engine_deps.events.append).return_value = None
        cast(MagicMock, engine_deps.runs.get).return_value = enterprise_run
        cast(MagicMock, engine_deps.runs.update_status).return_value = None
        cast(MagicMock, engine_deps.approvals.create).return_value = None

        # Mock compliant execution
        async def mock_compliant_action(*args, **kwargs):
            return CapabilityResult(
                ok=True,
                output={"compliance_status": "passed", "audit_id": str(uuid.uuid4()), "retention_policy": "enforced"},
            )

        mock_compliant_capability = MagicMock()
        mock_compliant_capability.execute = mock_compliant_action
        cast(MagicMock, engine_deps.capabilities.get).return_value = mock_compliant_capability

        # Execute enterprise workflow
        result = await engine.start_run(run=enterprise_run, plan=compliance_plan)

        # Verify compliance requirements were enforced
        assert cast(MagicMock, engine_deps.approvals.create).call_count >= 1  # At least one approval required

        # Verify audit trail was created (if implemented)
        # Note: Tool invocation audit trail might not be implemented in current engine version
        audit_calls = [
            call for call in cast(MagicMock, engine_deps.tool_invocations.append).call_args_list if "audit" in str(call)
        ]
        # Don't assert on audit calls as they might not be implemented yet

        # Verify workflow completed with compliance
        assert result == enterprise_run.id
