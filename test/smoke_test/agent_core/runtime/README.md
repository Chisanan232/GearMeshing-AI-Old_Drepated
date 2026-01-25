# LangGraph Runtime Engine Smoke Tests

This directory contains comprehensive smoke tests for the LangGraph-based runtime engine with AI agent workflows.

## Overview

The smoke tests verify the complete AI agent workflow execution using real AI models while mocking all other dependencies (database, cache, etc.). These tests follow the same design patterns as other smoke tests in the project.

## Test Files

### 1. `test_engine_e2e.py` - Core Runtime Engine Tests
**Normal Cases:**
- `test_complete_workflow_with_openai` - Complete workflow from planning to execution
- `test_multi_step_workflow_with_anthropic` - Multi-step workflow with Anthropic model
- `test_workflow_with_google_model` - Workflow execution with Google model
- `test_concurrent_workflows` - Concurrent execution of multiple workflows

**Edge Cases:**
- `test_workflow_with_capability_failure` - Handling when capabilities fail
- `test_workflow_with_approval_required` - Human-in-the-loop approval workflows
- `test_workflow_with_invalid_plan` - Invalid plan structure handling
- `test_workflow_with_empty_plan` - Empty plan handling

**State Management:**
- `test_workflow_checkpointing` - State checkpointing and resumption
- `test_workflow_event_logging` - Comprehensive event logging

### 2. `test_engine_advanced_e2e.py` - Advanced Workflow Tests
**Advanced Scenarios:**
- `test_long_running_workflow` - Long-running workflows with multiple phases
- `test_workflow_with_retry_mechanism` - Automatic retry on transient failures
- `test_workflow_with_dynamic_adaptation` - Workflows that adapt based on results
- `test_workflow_with_resource_constraints` - Execution under resource constraints
- `test_workflow_with_external_dependencies` - Workflows with external system dependencies

### 3. `test_engine_integration_e2e.py` - Real-World Integration Tests
**Integration Patterns:**
- `test_data_science_pipeline_workflow` - Complete data science pipeline
- `test_web_automation_workflow` - Web automation and scraping
- `test_human_in_the_loop_workflow` - Human-in-the-loop with approvals
- `test_multi_agent_collaboration_workflow` - Multi-agent collaboration
- `test_enterprise_workflow_compliance` - Enterprise compliance and audit

## Test Architecture

### Base Test Suite
All tests inherit from `BaseAIWorkflowTestSuite` which provides:

```python
class BaseAIWorkflowTestSuite:
    @pytest.fixture
    def mock_repositories(self) -> Dict[str, AsyncMock]:
        """Mock all repository dependencies."""
        
    @pytest.fixture
    def mock_capabilities(self) -> MagicMock:
        """Mock capabilities registry with realistic capabilities."""
        
    @pytest.fixture
    def mock_policy(self) -> GlobalPolicy:
        """Mock global policy for testing."""
        
    @pytest.fixture
    def sample_agent_run(self) -> AgentRun:
        """Sample agent run for testing."""
        
    @pytest.fixture
    def engine_deps(self) -> EngineDeps:
        """Create engine dependencies for testing."""
```

### Test Database Setup
Tests use a real SQLite database with model configurations:

```python
@pytest.fixture(scope="session")
def test_database():
    """Create a test database with model configurations for smoke tests."""
    # Creates SQLite in-memory database
    # Inserts model configurations for OpenAI, Anthropic, Google
    # Returns database URL for use in tests
```

### Real AI Model Integration
Tests use real AI models via `async_create_model_for_role`:

```python
# Create real AI model using settings from dotenv
thought_model = await async_create_model_for_role("assistant")

# Update engine deps with thought model
engine_deps = EngineDeps(
    # ... other dependencies
    thought_model=thought_model,
    # ...
)
```

## Test Categories

### Normal Cases
These tests verify standard workflow execution:
- Complete workflows from planning to completion
- Multi-step workflows with thought and action steps
- Different AI provider support (OpenAI, Anthropic, Google)
- Concurrent workflow execution

### Edge Cases
These tests verify error handling and edge cases:
- Capability failures and recovery
- Human-in-the-loop approval workflows
- Invalid plan structures
- Empty plans
- Resource constraints
- External dependencies

### Integration Scenarios
These tests verify real-world use cases:
- Data science pipelines
- Web automation
- Multi-agent collaboration
- Enterprise compliance
- Long-running workflows

## Running Tests

### Prerequisites
1. Configure API keys in `test/.env`:
   ```bash
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

### Running All Tests
```bash
# Run all runtime engine smoke tests
pytest test/smoke_test/agent_core/runtime/ -v

# Run with coverage
pytest test/smoke_test/agent_core/runtime/ -v --cov=gearmeshing_ai.agent_core.runtime
```

### Running Specific Test Categories
```bash
# Run only core tests
pytest test/smoke_test/agent_core/runtime/test_engine_e2e.py -v

# Run only advanced tests
pytest test/smoke_test/agent_core/runtime/test_engine_advanced_e2e.py -v

# Run only integration tests
pytest test/smoke_test/agent_core/runtime/test_engine_integration_e2e.py -v
```

### Running with Specific AI Provider
```bash
# Run only OpenAI tests
pytest test/smoke_test/agent_core/runtime/ -v -k "openai"

# Run only Anthropic tests
pytest test/smoke_test/agent_core/runtime/ -v -k "anthropic"

# Run only Google tests
pytest test/smoke_test/agent_core/runtime/ -v -k "google"
```

## Test Markers

Tests use pytest markers for categorization:

```python
@pytest.mark.asyncio
@pytest.mark.smoke_ai
async def test_example(self):
    """Test marked as smoke test with AI dependencies."""
```

## Mock Strategy

### Repository Mocks
All database repositories are mocked:
```python
mock_repositories = {
    "runs": AsyncMock(),
    "events": AsyncMock(),
    "approvals": AsyncMock(),
    "checkpoints": AsyncMock(),
    "tool_invocations": AsyncMock(),
    "usage": AsyncMock(),
}
```

### Capability Mocks
Capabilities are mocked with realistic interfaces:
```python
mock_capabilities.list_all.return_value = [
    {
        "name": "read_file",
        "description": "Read a file from the filesystem",
        "parameters": {"file_path": "string"},
    },
    # ... more capabilities
]
```

### Policy Mocks
Policies are mocked to control behavior:
```python
# Allow all actions
mock_decision.block = False
mock_decision.require_approval = False

# Require approval for sensitive actions
mock_decision.require_approval = True
mock_decision.risk = RiskLevel.high
```

## Expected Test Results

### Successful Execution
- Tests should complete successfully when API keys are configured
- Events should be logged properly
- Mock repositories should be called appropriately
- Workflow should progress through all steps

### Expected Skips
Tests will be skipped when:
- Required API keys are not configured
- External dependencies are unavailable
- Test environment doesn't support certain features

### Expected Failures
Tests should fail when:
- Invalid plan structures are provided
- Critical errors occur in workflow execution
- Required dependencies are missing

## Debugging Tips

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
pytest test/smoke_test/agent_core/runtime/ -v -s
```

### Check API Key Configuration
```python
from test.settings import test_settings
print(f"OpenAI: {bool(test_settings.ai_provider.openai.api_key)}")
print(f"Anthropic: {bool(test_settings.ai_provider.anthropic.api_key)}")
print(f"Google: {bool(test_settings.ai_provider.google.api_key)}")
```

### Inspect Mock Calls
```python
# Check what was called
mock_repository.create.call_args_list
mock_repository.update_status.call_args_list

# Check call counts
assert mock_events.append.call_count > 0
```

## Best Practices

### Test Structure
1. **Arrange**: Setup test data, mocks, and configuration
2. **Act**: Execute the workflow
3. **Assert**: Verify expected behavior and side effects

### Mock Usage
- Use realistic mock responses
- Verify mock calls were made appropriately
- Don't over-mock - only mock external dependencies

### Error Testing
- Test both success and failure scenarios
- Verify proper error handling and recovery
- Test edge cases and boundary conditions

### Event Verification
- Always verify events are logged
- Check event types and payloads
- Ensure event ordering is correct

## Contributing

When adding new tests:

1. Follow the existing test structure and patterns
2. Use appropriate test markers (`@pytest.mark.smoke_ai`)
3. Add comprehensive assertions
4. Include both positive and negative test cases
5. Document the test purpose and scenarios
6. Update this README if adding new test categories

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are configured in `test/.env`
   - Check environment variable export
   - Verify API key validity

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

3. **Mock Configuration**
   - Ensure mocks are properly configured
   - Check mock return values
   - Verify mock call expectations

4. **Async Test Issues**
   - Ensure `@pytest.mark.asyncio` is used
   - Check async/await usage
   - Verify proper coroutine handling

### Getting Help

1. Check the test output for specific error messages
2. Enable debug logging for more detailed output
3. Run individual tests to isolate issues
4. Check the project documentation for additional context
