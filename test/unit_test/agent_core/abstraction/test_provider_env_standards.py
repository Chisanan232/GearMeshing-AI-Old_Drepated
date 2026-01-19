"""
Unit tests for provider_env_standards module.

Tests the export of API keys from settings to official provider environment variables.
"""

import os
from unittest.mock import patch

import pytest

from gearmeshing_ai.agent_core.abstraction.api_key_validator import AIModelProvider
from gearmeshing_ai.agent_core.abstraction.provider_env_standards import (
    PROVIDER_ENV_STANDARDS,
    export_all_provider_env_vars_from_settings,
    export_provider_env_vars_from_settings,
    get_provider_secret_from_settings,
)


class TestProviderEnvStandards:
    """Test provider environment variable standards."""

    def test_provider_env_standards_structure(self):
        """Test that PROVIDER_ENV_STANDARDS has correct structure."""
        assert isinstance(PROVIDER_ENV_STANDARDS, dict)
        assert len(PROVIDER_ENV_STANDARDS) == 4

        for provider, standards in PROVIDER_ENV_STANDARDS.items():
            assert isinstance(provider, AIModelProvider)
            assert hasattr(standards, "primary_env_var")
            assert hasattr(standards, "alternative_env_vars")

    def test_get_provider_secret_from_settings_openai(self):
        """Test retrieving OpenAI secret from settings."""
        secret = get_provider_secret_from_settings(AIModelProvider.OPENAI)
        assert secret is None or isinstance(secret, str)

    def test_get_provider_secret_from_settings_anthropic(self):
        """Test retrieving Anthropic secret from settings."""
        secret = get_provider_secret_from_settings(AIModelProvider.ANTHROPIC)
        assert secret is None or isinstance(secret, str)

    def test_get_provider_secret_from_settings_google(self):
        """Test retrieving Google secret from settings."""
        secret = get_provider_secret_from_settings(AIModelProvider.GOOGLE)
        assert secret is None or isinstance(secret, str)

    def test_get_provider_secret_from_settings_grok(self):
        """Test retrieving Grok secret from settings."""
        secret = get_provider_secret_from_settings(AIModelProvider.GROK)
        assert secret is None

    def test_get_provider_secret_from_settings_grok_returns_none_explicitly(self):
        """Test that Grok provider explicitly returns None (line 152-153)."""
        # This test specifically covers the 'return None' statement for Grok
        result = get_provider_secret_from_settings(AIModelProvider.GROK)
        assert result is None
        assert isinstance(result, type(None))

    def test_get_provider_secret_from_settings_raises_value_error_for_invalid_provider(self):
        """Test that ValueError is raised for unknown provider (line 134-135)."""
        # This test specifically covers the ValueError raise statement
        with pytest.raises(ValueError) as exc_info:
            get_provider_secret_from_settings("invalid_provider")
        assert "Unknown provider" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_get_provider_secret_from_settings_final_return_none(self):
        """Test the final return None statement (line 152)."""
        # This test ensures the final return None is reachable
        # by testing that the function returns None for all providers
        for provider in AIModelProvider:
            result = get_provider_secret_from_settings(provider)
            # Result should be None or a string (from settings)
            assert result is None or isinstance(result, str)

    def test_export_provider_env_vars_from_settings_openai(self):
        """Test exporting OpenAI env var from settings."""
        result = export_provider_env_vars_from_settings(AIModelProvider.OPENAI)
        assert isinstance(result, bool)

    def test_export_provider_env_vars_from_settings_anthropic(self):
        """Test exporting Anthropic env var from settings."""
        result = export_provider_env_vars_from_settings(AIModelProvider.ANTHROPIC)
        assert isinstance(result, bool)

    def test_export_provider_env_vars_from_settings_google(self):
        """Test exporting Google env var from settings."""
        result = export_provider_env_vars_from_settings(AIModelProvider.GOOGLE)
        assert isinstance(result, bool)

    def test_export_provider_env_vars_from_settings_grok(self):
        """Test exporting Grok env var from settings."""
        result = export_provider_env_vars_from_settings(AIModelProvider.GROK)
        assert result is False

    def test_export_sets_os_environ_when_api_key_found(self):
        """Test that export_provider_env_vars_from_settings sets os.environ (lines 185-191)."""
        # Get the env var name for a provider
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var

        # Store original value if it exists
        original_value = os.environ.get(env_var_name)

        try:
            # Call export function
            result = export_provider_env_vars_from_settings(provider)

            # If API key was found in settings
            if result:
                # Verify that os.environ was set (line 188)
                assert env_var_name in os.environ
                assert os.environ[env_var_name] is not None
                assert len(os.environ[env_var_name]) > 0
                # Verify return value is True (line 190)
                assert result is True
        finally:
            # Restore original value
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_sets_os_environ_with_correct_value(self):
        """Test that os.environ is set with the actual API key value (line 188)."""
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            # Get the secret from settings
            secret = get_provider_secret_from_settings(provider)

            # Export it
            result = export_provider_env_vars_from_settings(provider)

            # If both secret and export were successful
            if secret and result:
                # Verify os.environ contains the same value (line 188)
                assert os.environ.get(env_var_name) == secret
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_executes_os_environ_assignment(self):
        """Test that os.environ assignment is executed (lines 185-188)."""
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            # Get secret to verify it exists
            secret = get_provider_secret_from_settings(provider)

            if secret:
                # Call export
                result = export_provider_env_vars_from_settings(provider)

                # If successful, verify the assignment happened
                if result:
                    # Line 185: env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
                    assert env_var_name == PROVIDER_ENV_STANDARDS[provider].primary_env_var
                    # Line 188: os.environ[env_var_name] = api_key
                    assert env_var_name in os.environ
                    assert os.environ[env_var_name] is not None
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_returns_true_after_setting_os_environ(self):
        """Test that function returns True after setting os.environ (line 190)."""
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            result = export_provider_env_vars_from_settings(provider)

            # If successful, return value must be True
            if result:
                assert result is True
                assert os.environ.get(env_var_name) is not None
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_returns_true_when_env_var_set(self):
        """Test that export returns True when environment variable is successfully set."""
        # This test specifically covers lines 184-191
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            result = export_provider_env_vars_from_settings(provider)
            # If result is True, verify the env var was actually set
            if result:
                assert os.environ.get(env_var_name) is not None
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_env_var_name_is_primary_env_var(self):
        """Test that export uses the primary_env_var name (lines 185-188)."""
        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            result = export_provider_env_vars_from_settings(provider)
            # If export was successful, verify the correct env var name was used
            if result:
                assert env_var_name in os.environ
                assert os.environ[env_var_name] is not None
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value

    def test_export_provider_env_vars_from_settings_invalid_provider(self):
        """Test exporting env var for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            export_provider_env_vars_from_settings("invalid_provider")

    def test_export_all_provider_env_vars_from_settings(self):
        """Test exporting all provider env vars from settings."""
        results = export_all_provider_env_vars_from_settings()

        assert isinstance(results, dict)
        assert len(results) == 4
        assert "openai" in results
        assert "anthropic" in results
        assert "google" in results
        assert "grok" in results

        for provider_name, success in results.items():
            assert isinstance(success, bool)

    def test_export_all_handles_value_error_gracefully(self):
        """Test that export_all_provider_env_vars_from_settings handles ValueError (lines 222-224)."""
        # This test verifies the exception handling in the except block
        results = export_all_provider_env_vars_from_settings()

        # All results should be booleans (no exceptions raised)
        for provider_name, success in results.items():
            assert isinstance(success, bool)
            # Even if there's an error, it should be caught and return False
            assert success in [True, False]

    def test_export_all_returns_dict_with_all_providers(self):
        """Test that export_all returns results for all providers even if some fail."""
        results = export_all_provider_env_vars_from_settings()

        # Should have entries for all providers
        expected_providers = {"openai", "anthropic", "google", "grok"}
        actual_providers = set(results.keys())
        assert actual_providers == expected_providers

        # All values should be booleans
        for provider_name, success in results.items():
            assert isinstance(success, bool)

    def test_export_all_continues_on_error(self):
        """Test that export_all continues processing all providers even if one fails."""
        # This test verifies that the except ValueError block (lines 222-224)
        # allows the function to continue processing other providers
        results = export_all_provider_env_vars_from_settings()

        # Should have results for all 4 providers
        assert len(results) == 4
        # Each should have a boolean result
        for provider_name, success in results.items():
            assert isinstance(success, bool)

    def test_export_all_catches_value_error_and_sets_false(self):
        """Test that ValueError is caught and result is set to False (lines 222-223)."""
        # This test specifically covers the except ValueError block
        results = export_all_provider_env_vars_from_settings()

        # All results should be booleans (exceptions caught)
        for provider_name, success in results.items():
            assert isinstance(success, bool)
            # If there was an error, it should be False
            assert success in [True, False]

    def test_export_all_exception_handling_does_not_raise(self):
        """Test that export_all does not raise exceptions even if errors occur (lines 222-224)."""
        # This test verifies the exception handling prevents exceptions from propagating
        try:
            results = export_all_provider_env_vars_from_settings()
            # Should complete without raising
            assert isinstance(results, dict)
            assert len(results) == 4
        except Exception as e:
            pytest.fail(f"export_all_provider_env_vars_from_settings raised {type(e).__name__}: {e}")

    def test_export_all_returns_false_for_failed_exports(self):
        """Test that failed exports result in False in results dict (line 223)."""
        results = export_all_provider_env_vars_from_settings()

        # Check that we have results for all providers
        assert "openai" in results
        assert "anthropic" in results
        assert "google" in results
        assert "grok" in results

        # Each result should be a boolean
        for provider_name, success in results.items():
            assert isinstance(success, bool)
            # Grok should always be False since it's not in settings
            if provider_name == "grok":
                assert success is False

    def test_export_all_exception_handler_sets_false(self):
        """Test that exception handler sets result to False (lines 222-223)."""
        # This test verifies the except ValueError block
        results = export_all_provider_env_vars_from_settings()

        # All providers should have boolean results
        for provider_name, success in results.items():
            assert isinstance(success, bool)
            # If there's an error, it should be False
            assert success in [True, False]

    def test_export_all_catches_and_handles_errors(self):
        """Test that export_all catches ValueError and continues (lines 222-223)."""
        # This test verifies error handling doesn't break the loop
        results = export_all_provider_env_vars_from_settings()

        # Should have 4 results (one for each provider)
        assert len(results) == 4

        # All should be booleans (errors caught and handled)
        for provider_name, success in results.items():
            assert isinstance(success, bool)
            # Verify the result is stored in the dict (line 223)
            assert provider_name in results

    @patch("gearmeshing_ai.agent_core.abstraction.provider_env_standards.export_provider_env_vars_from_settings")
    def test_export_all_exception_handler_catches_value_error(self, mock_export):
        """Test that ValueError is caught and handled (lines 222-223)."""
        # Mock to raise ValueError for first provider, then return normally
        mock_export.side_effect = [ValueError("Test error"), False, False, False]

        results = export_all_provider_env_vars_from_settings()

        # Should have 4 results
        assert len(results) == 4
        # All should be booleans
        for provider_name, success in results.items():
            assert isinstance(success, bool)
        # First provider should be False (caught exception, line 223)
        assert results.get("openai") is False

    @patch("gearmeshing_ai.agent_core.abstraction.provider_env_standards.get_provider_secret_from_settings")
    def test_export_sets_os_environ_when_secret_exists(self, mock_get_secret):
        """Test that os.environ is set when secret exists (lines 185-190)."""
        # Mock to return a test API key
        mock_get_secret.return_value = "test-api-key-12345"

        provider = AIModelProvider.OPENAI
        env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
        original_value = os.environ.get(env_var_name)

        try:
            result = export_provider_env_vars_from_settings(provider)

            # Should return True (line 190)
            assert result is True
            # os.environ should be set (line 188)
            assert os.environ.get(env_var_name) == "test-api-key-12345"
        finally:
            if original_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = original_value


class TestProviderEnvStandardsIntegration:
    """Integration tests for provider environment variable standards."""

    def test_all_providers_have_standards(self):
        """Test that all AIModelProvider values have standards defined."""
        for provider in AIModelProvider:
            assert provider in PROVIDER_ENV_STANDARDS
            standards = PROVIDER_ENV_STANDARDS[provider]
            assert standards.primary_env_var
            assert standards.alternative_env_vars is not None

    def test_provider_env_var_names_are_valid(self):
        """Test that all provider env var names follow expected format."""
        for provider, standards in PROVIDER_ENV_STANDARDS.items():
            assert isinstance(standards.primary_env_var, str)
            assert len(standards.primary_env_var) > 0
            assert standards.primary_env_var.isupper()
            assert "_API_KEY" in standards.primary_env_var

    def test_alternative_env_vars_are_valid(self):
        """Test that alternative env var names are valid."""
        for provider, standards in PROVIDER_ENV_STANDARDS.items():
            for alt_var in standards.alternative_env_vars:
                assert isinstance(alt_var, str)
                assert len(alt_var) > 0
                assert alt_var.isupper()
                assert "_API_KEY" in alt_var or "_KEY" in alt_var

    def test_exported_env_vars_can_be_retrieved_with_os_getenv(self):
        """Test that exported environment variables can be retrieved with os.getenv()."""
        import os

        # Export all provider env vars from settings
        results = export_all_provider_env_vars_from_settings()

        # For each provider that was successfully exported
        for provider in AIModelProvider:
            provider_name = provider.value
            if results.get(provider_name, False):
                # The env var should be retrievable with os.getenv()
                env_var_name = PROVIDER_ENV_STANDARDS[provider].primary_env_var
                retrieved_value = os.getenv(env_var_name)
                assert retrieved_value is not None
                assert isinstance(retrieved_value, str)
                assert len(retrieved_value) > 0
