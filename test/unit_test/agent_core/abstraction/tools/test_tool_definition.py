"""Unit tests for ToolDefinition Pydantic model.

Tests for the ToolDefinition class including validation, serialization,
and utility methods.
"""

import json
from typing import Any, Dict, Optional
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from gearmeshing_ai.agent_core.abstraction.tools.definitions import (
    CommandRunInput,
    CommandRunOutput,
    FileReadInput,
    FileReadOutput,
    FileWriteInput,
    FileWriteOutput,
    ToolDefinition,
)


class TestToolDefinitionCreation:
    """Tests for ToolDefinition creation and initialization."""

    def test_create_tool_definition_with_all_fields(self) -> None:
        """Test creating a ToolDefinition with all fields."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=dummy_handler,
        )

        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.input_schema == FileReadInput
        assert tool.output_schema == FileReadOutput
        assert tool.handler == dummy_handler

    def test_create_tool_definition_without_handler(self) -> None:
        """Test creating a ToolDefinition without handler."""
        tool = ToolDefinition(
            name="write_file",
            description="Write a file",
            input_schema=FileWriteInput,
            output_schema=FileWriteOutput,
        )

        assert tool.name == "write_file"
        assert tool.description == "Write a file"
        assert tool.input_schema == FileWriteInput
        assert tool.output_schema == FileWriteOutput
        assert tool.handler is None

    def test_create_tool_definition_with_none_handler(self) -> None:
        """Test creating a ToolDefinition with explicit None handler."""
        tool = ToolDefinition(
            name="run_command",
            description="Run a command",
            input_schema=CommandRunInput,
            output_schema=CommandRunOutput,
            handler=None,
        )

        assert tool.handler is None

    def test_tool_definition_validation_missing_required_field(self) -> None:
        """Test that ToolDefinition validates required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition(  # type: ignore[call-arg]
                name="test",
                # Missing description
                input_schema=FileReadInput,
                output_schema=FileReadOutput,
            )

        assert "description" in str(exc_info.value)

    def test_tool_definition_validation_invalid_name_type(self) -> None:
        """Test that ToolDefinition validates field types."""
        with pytest.raises(ValidationError):
            ToolDefinition(
                name=123,  # Should be string
                description="Test",
                input_schema=FileReadInput,
                output_schema=FileReadOutput,
            )

    def test_tool_definition_validation_invalid_input_schema_type(self) -> None:
        """Test that ToolDefinition validates input_schema type."""
        with pytest.raises(ValidationError):
            ToolDefinition(
                name="test",
                description="Test",
                input_schema="not_a_class",  # Should be Type[BaseModel]
                output_schema=FileReadOutput,
            )


class TestToolDefinitionSerialization:
    """Tests for ToolDefinition serialization methods."""

    def test_to_dict_returns_basic_info(self) -> None:
        """Test to_dict returns name, description, and input schema."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file from filesystem",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        result = tool.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "read_file"
        assert result["description"] == "Read a file from filesystem"
        assert "input_schema" in result
        assert isinstance(result["input_schema"], dict)

    def test_to_dict_includes_input_schema_json(self) -> None:
        """Test that to_dict includes input schema as JSON schema."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        result = tool.to_dict()
        input_schema = result["input_schema"]

        # Verify it's a valid JSON schema
        assert "type" in input_schema
        assert "properties" in input_schema
        assert "file_path" in input_schema["properties"]
        assert "encoding" in input_schema["properties"]

    def test_to_full_dict_returns_complete_info(self) -> None:
        """Test to_full_dict returns all information including output schema."""
        tool = ToolDefinition(
            name="write_file",
            description="Write a file",
            input_schema=FileWriteInput,
            output_schema=FileWriteOutput,
        )

        result = tool.to_full_dict()

        assert result["name"] == "write_file"
        assert result["description"] == "Write a file"
        assert "input_schema" in result
        assert "output_schema" in result
        assert isinstance(result["input_schema"], dict)
        assert isinstance(result["output_schema"], dict)

    def test_to_full_dict_includes_output_schema_json(self) -> None:
        """Test that to_full_dict includes output schema as JSON schema."""
        tool = ToolDefinition(
            name="write_file",
            description="Write a file",
            input_schema=FileWriteInput,
            output_schema=FileWriteOutput,
        )

        result = tool.to_full_dict()
        output_schema = result["output_schema"]

        # Verify it's a valid JSON schema
        assert "type" in output_schema
        assert "properties" in output_schema
        assert "success" in output_schema["properties"]
        assert "file_path" in output_schema["properties"]
        assert "bytes_written" in output_schema["properties"]

    def test_model_dump_returns_dictionary(self) -> None:
        """Test that model_dump returns a dictionary representation."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=dummy_handler,
        )

        result = tool.model_dump(exclude={"input_schema", "output_schema", "handler"})

        assert isinstance(result, dict)
        assert result["name"] == "read_file"
        assert result["description"] == "Read a file"

    def test_model_dump_with_excluded_fields(self) -> None:
        """Test that model_dump works when excluding non-serializable fields."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=None,
        )

        # Exclude fields that can't be serialized
        result = tool.model_dump(exclude={"input_schema", "output_schema", "handler"})

        assert isinstance(result, dict)
        assert result["name"] == "read_file"
        assert result["description"] == "Read a file"
        assert "input_schema" not in result
        assert "output_schema" not in result
        assert "handler" not in result


class TestToolDefinitionSchemaAccess:
    """Tests for schema access methods."""

    def test_get_input_schema_json_returns_json_schema(self) -> None:
        """Test get_input_schema_json returns input schema as JSON schema."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        result = tool.get_input_schema_json()

        assert isinstance(result, dict)
        assert "type" in result
        assert "properties" in result
        assert "file_path" in result["properties"]
        assert "encoding" in result["properties"]

    def test_get_output_schema_json_returns_json_schema(self) -> None:
        """Test get_output_schema_json returns output schema as JSON schema."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        result = tool.get_output_schema_json()

        assert isinstance(result, dict)
        assert "type" in result
        assert "properties" in result
        assert "success" in result["properties"]
        assert "content" in result["properties"]
        assert "error" in result["properties"]
        assert "file_path" in result["properties"]
        assert "size_bytes" in result["properties"]

    def test_get_input_schema_json_for_command_run(self) -> None:
        """Test get_input_schema_json for CommandRunInput."""
        tool = ToolDefinition(
            name="run_command",
            description="Run a command",
            input_schema=CommandRunInput,
            output_schema=CommandRunOutput,
        )

        result = tool.get_input_schema_json()

        assert "command" in result["properties"]
        assert "cwd" in result["properties"]
        assert "timeout" in result["properties"]
        assert "shell" in result["properties"]

    def test_get_output_schema_json_for_command_run(self) -> None:
        """Test get_output_schema_json for CommandRunOutput."""
        tool = ToolDefinition(
            name="run_command",
            description="Run a command",
            input_schema=CommandRunInput,
            output_schema=CommandRunOutput,
        )

        result = tool.get_output_schema_json()

        assert "success" in result["properties"]
        assert "exit_code" in result["properties"]
        assert "stdout" in result["properties"]
        assert "stderr" in result["properties"]
        assert "command" in result["properties"]
        assert "duration_seconds" in result["properties"]


class TestToolDefinitionHandlerManagement:
    """Tests for handler management methods."""

    def test_set_handler_assigns_handler(self) -> None:
        """Test set_handler assigns a handler function."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        assert tool.handler is None

        tool.set_handler(dummy_handler)

        assert tool.handler == dummy_handler

    def test_set_handler_overwrites_existing_handler(self) -> None:
        """Test set_handler overwrites an existing handler."""
        def handler1(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        def handler2(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=False, file_path="", error="Error")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=handler1,
        )

        assert tool.handler == handler1

        tool.set_handler(handler2)

        assert tool.handler == handler2

    def test_has_handler_returns_true_when_handler_set(self) -> None:
        """Test has_handler returns True when handler is set."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=dummy_handler,
        )

        assert tool.has_handler() is True

    def test_has_handler_returns_false_when_handler_not_set(self) -> None:
        """Test has_handler returns False when handler is not set."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        assert tool.has_handler() is False

    def test_has_handler_returns_false_when_handler_is_none(self) -> None:
        """Test has_handler returns False when handler is explicitly None."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=None,
        )

        assert tool.has_handler() is False

    def test_has_handler_after_setting_handler(self) -> None:
        """Test has_handler returns True after setting handler."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        assert tool.has_handler() is False

        tool.set_handler(dummy_handler)

        assert tool.has_handler() is True


class TestToolDefinitionFieldDescriptions:
    """Tests for field descriptions and metadata."""

    def test_field_descriptions_via_model_fields(self) -> None:
        """Test that field descriptions are accessible via model fields."""
        fields = ToolDefinition.model_fields

        assert fields["name"].description == "Unique identifier for the tool"
        assert fields["description"].description is not None
        assert "Human-readable description" in fields["description"].description
        assert fields["input_schema"].description is not None
        assert "Pydantic model class for input validation" in fields["input_schema"].description
        assert fields["output_schema"].description is not None
        assert "Pydantic model class for output" in fields["output_schema"].description

    def test_handler_field_is_optional_via_model_fields(self) -> None:
        """Test that handler field is optional."""
        fields = ToolDefinition.model_fields

        # handler should be optional (default=None)
        assert fields["handler"].is_required() is False

    def test_required_fields_are_enforced_via_model_fields(self) -> None:
        """Test that required fields are enforced."""
        fields = ToolDefinition.model_fields

        assert fields["name"].is_required() is True
        assert fields["description"].is_required() is True
        assert fields["input_schema"].is_required() is True
        assert fields["output_schema"].is_required() is True


class TestToolDefinitionEquality:
    """Tests for tool definition equality and comparison."""

    def test_two_identical_tools_are_equal(self) -> None:
        """Test that two identical tool definitions are equal."""
        tool1 = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        tool2 = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        assert tool1 == tool2

    def test_tools_with_different_names_are_not_equal(self) -> None:
        """Test that tools with different names are not equal."""
        tool1 = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        tool2 = ToolDefinition(
            name="write_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        assert tool1 != tool2

    def test_tools_with_different_handlers_are_not_equal(self) -> None:
        """Test that tools with different handlers are not equal."""
        def handler1(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="", content="")

        def handler2(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=False, file_path="", error="Error")

        tool1 = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=handler1,
        )

        tool2 = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
            handler=handler2,
        )

        assert tool1 != tool2


class TestToolDefinitionIntegration:
    """Integration tests for ToolDefinition."""

    def test_tool_definition_with_all_utility_methods(self) -> None:
        """Test using all utility methods together."""
        def dummy_handler(input_data: FileReadInput) -> FileReadOutput:
            return FileReadOutput(success=True, file_path="test.txt", content="data")

        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        # Initially no handler
        assert not tool.has_handler()

        # Set handler
        tool.set_handler(dummy_handler)
        assert tool.has_handler()

        # Get schemas
        input_schema = tool.get_input_schema_json()
        output_schema = tool.get_output_schema_json()
        assert input_schema is not None
        assert output_schema is not None

        # Convert to dictionaries
        basic_dict = tool.to_dict()
        full_dict = tool.to_full_dict()
        assert "input_schema" in basic_dict
        assert "output_schema" in full_dict

    def test_multiple_tools_with_different_schemas(self) -> None:
        """Test creating multiple tools with different input/output schemas."""
        read_tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        write_tool = ToolDefinition(
            name="write_file",
            description="Write a file",
            input_schema=FileWriteInput,
            output_schema=FileWriteOutput,
        )

        command_tool = ToolDefinition(
            name="run_command",
            description="Run a command",
            input_schema=CommandRunInput,
            output_schema=CommandRunOutput,
        )

        # Verify each tool has correct schemas
        assert read_tool.input_schema == FileReadInput
        assert write_tool.input_schema == FileWriteInput
        assert command_tool.input_schema == CommandRunInput

        assert read_tool.output_schema == FileReadOutput
        assert write_tool.output_schema == FileWriteOutput
        assert command_tool.output_schema == CommandRunOutput

    def test_tool_definition_copy_and_update(self) -> None:
        """Test copying and updating tool definitions."""
        original_tool = ToolDefinition(
            name="read_file",
            description="Read a file from filesystem",
            input_schema=FileReadInput,
            output_schema=FileReadOutput,
        )

        # Create a copy with updated description
        updated_tool = original_tool.model_copy(update={"description": "Read a file (updated)"})

        # Verify original is unchanged
        assert original_tool.description == "Read a file from filesystem"
        # Verify copy has updated value
        assert updated_tool.description == "Read a file (updated)"
        # Verify other fields are the same
        assert updated_tool.name == original_tool.name
        assert updated_tool.input_schema == original_tool.input_schema
        assert updated_tool.output_schema == original_tool.output_schema
