"""Unit tests for CloudTrain configuration manager."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from cloudtrain.config.manager import ConfigManager, ConfigurationError
from cloudtrain.config.settings import (
    AWSConfig,
    AzureConfig,
    CloudTrainSettings,
    GCPConfig,
    MockConfig,
)
from cloudtrain.enums import CloudProvider, LogLevel


class TestConfigurationError:
    """Test ConfigurationError exception class."""

    def test_configuration_error_creation(self) -> None:
        """Test creating ConfigurationError with message."""
        error: ConfigurationError = ConfigurationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_configuration_error_inheritance(self) -> None:
        """Test ConfigurationError inherits from Exception."""
        error: ConfigurationError = ConfigurationError("Test")
        assert isinstance(error, Exception)

    def test_configuration_error_without_message(self) -> None:
        """Test creating ConfigurationError without message."""
        error: ConfigurationError = ConfigurationError()
        assert isinstance(error, Exception)


class TestConfigManager:
    """Test ConfigManager class."""

    def test_init_with_settings(self) -> None:
        """Test initialization with pre-configured settings."""
        settings: CloudTrainSettings = CloudTrainSettings(log_level=LogLevel.DEBUG)

        with patch.object(ConfigManager, "_configure_logging") as mock_logging:
            manager: ConfigManager = ConfigManager(settings=settings)

        assert manager.settings == settings
        assert "programmatic" in manager.config_sources
        mock_logging.assert_called_once()

    def test_init_without_settings(self) -> None:
        """Test initialization without settings loads from environment."""
        with (
            patch.object(ConfigManager, "_load_settings") as mock_load,
            patch.object(ConfigManager, "_configure_logging") as mock_logging,
        ):
            mock_settings: CloudTrainSettings = CloudTrainSettings()
            mock_load.return_value = mock_settings

            manager: ConfigManager = ConfigManager()

            assert manager.settings == mock_settings
            mock_load.assert_called_once_with(None)
            mock_logging.assert_called_once()

    def test_init_with_config_file(self) -> None:
        """Test initialization with config file path."""
        config_file: str = "/path/to/config.yaml"

        with (
            patch.object(ConfigManager, "_load_settings") as mock_load,
            patch.object(ConfigManager, "_configure_logging") as mock_logging,
        ):
            mock_settings: CloudTrainSettings = CloudTrainSettings()
            mock_load.return_value = mock_settings

            manager: ConfigManager = ConfigManager(config_file=config_file)

            mock_load.assert_called_once_with(config_file)
            mock_logging.assert_called_once()

    def test_load_settings_environment_only(self):
        """Test loading settings from environment only."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []

        with patch(
            "cloudtrain.config.manager.CloudTrainSettings"
        ) as mock_settings_class:
            mock_settings = CloudTrainSettings()
            mock_settings_class.return_value = mock_settings

            result = manager._load_settings()

            assert result == mock_settings
            assert "environment" in manager.config_sources

    def test_load_settings_environment_validation_error(self):
        """Test handling validation error from environment settings."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []

        with patch(
            "cloudtrain.config.manager.CloudTrainSettings"
        ) as mock_settings_class:
            # Create a proper ValidationError by trying to create an invalid CloudTrainSettings
            def create_validation_error(*args, **kwargs):
                try:
                    CloudTrainSettings(log_level="invalid_level")
                except ValidationError as e:
                    raise e
                raise Exception("Should have raised ValidationError")

            mock_settings_class.side_effect = create_validation_error

            with pytest.raises(
                ConfigurationError, match="Invalid environment configuration"
            ):
                manager._load_settings()

    def test_load_settings_with_config_file(self):
        """Test loading settings with explicit config file."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []
        config_file = "/path/to/config.yaml"
        file_config = {"log_level": "debug"}

        with (
            patch(
                "cloudtrain.config.manager.CloudTrainSettings"
            ) as mock_settings_class,
            patch.object(manager, "_load_config_file") as mock_load_file,
        ):
            mock_settings = CloudTrainSettings()
            mock_settings_class.side_effect = [
                mock_settings,
                mock_settings,
            ]  # env, then file
            mock_load_file.return_value = file_config

            result = manager._load_settings(config_file)

            assert result == mock_settings
            assert "environment" in manager.config_sources
            assert f"file:{config_file}" in manager.config_sources
            mock_load_file.assert_called_once_with(config_file)

    def test_load_settings_config_file_validation_error(self):
        """Test handling validation error from config file."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []
        config_file = "/path/to/config.yaml"
        file_config = {"invalid_field": "value"}

        with (
            patch(
                "cloudtrain.config.manager.CloudTrainSettings"
            ) as mock_settings_class,
            patch.object(manager, "_load_config_file") as mock_load_file,
        ):
            mock_settings = CloudTrainSettings()

            # Create a ValidationError to be raised
            validation_error = None
            try:
                CloudTrainSettings(log_level="invalid_level")
            except ValidationError as e:
                validation_error = e

            mock_settings_class.side_effect = [
                mock_settings,  # env settings
                validation_error,
            ]
            mock_load_file.return_value = file_config

            with pytest.raises(ConfigurationError, match="Invalid file configuration"):
                manager._load_settings(config_file)

    def test_load_settings_config_file_not_found(self):
        """Test loading settings when config file returns None."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []
        config_file = "/path/to/nonexistent.yaml"

        with (
            patch(
                "cloudtrain.config.manager.CloudTrainSettings"
            ) as mock_settings_class,
            patch.object(manager, "_load_config_file") as mock_load_file,
        ):
            mock_settings = CloudTrainSettings()
            mock_settings_class.return_value = mock_settings
            mock_load_file.return_value = None

            result = manager._load_settings(config_file)

            assert result == mock_settings
            assert "environment" in manager.config_sources
            assert f"file:{config_file}" not in manager.config_sources

    @patch("cloudtrain.config.manager.Path")
    def test_load_settings_default_files_found(self, mock_path_class):
        """Test loading settings from default config files."""
        manager = ConfigManager.__new__(ConfigManager)
        manager.config_sources = []

        # Mock Path behavior
        mock_config_path = Mock()
        mock_config_path.exists.return_value = True
        mock_path_class.cwd.return_value.__truediv__.return_value = mock_config_path
        mock_path_class.home.return_value.__truediv__.return_value.__truediv__.return_value = (
            Mock()
        )
        mock_path_class.home.return_value.__truediv__.return_value.__truediv__.return_value.exists.return_value = (
            False
        )

        file_config = {"log_level": "debug"}

        with (
            patch(
                "cloudtrain.config.manager.CloudTrainSettings"
            ) as mock_settings_class,
            patch.object(manager, "_load_config_file") as mock_load_file,
        ):
            mock_settings = CloudTrainSettings()
            mock_settings_class.return_value = mock_settings
            mock_load_file.return_value = file_config

            result = manager._load_settings()

            assert result == mock_settings
            assert "environment" in manager.config_sources
            assert any("file:" in source for source in manager.config_sources)

    def test_load_config_file_nonexistent(self):
        """Test loading non-existent config file."""
        manager = ConfigManager.__new__(ConfigManager)
        config_file = Path("/nonexistent/config.yaml")

        with patch("cloudtrain.config.manager.logger") as mock_logger:
            result = manager._load_config_file(config_file)

        assert result is None
        mock_logger.warning.assert_called_once()

    def test_load_config_file_yaml(self):
        """Test loading YAML config file."""
        manager = ConfigManager.__new__(ConfigManager)
        config_data = {"log_level": "debug", "mock": {"enabled": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            result = manager._load_config_file(config_file)
            assert result == config_data
        finally:
            config_file.unlink()

    def test_load_config_file_json(self):
        """Test loading JSON config file."""
        manager = ConfigManager.__new__(ConfigManager)
        config_data = {"log_level": "debug", "mock": {"enabled": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)

        try:
            result = manager._load_config_file(config_file)
            assert result == config_data
        finally:
            config_file.unlink()

    def test_load_config_file_unsupported_format(self):
        """Test loading config file with unsupported format."""
        manager = ConfigManager.__new__(ConfigManager)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            config_file = Path(f.name)

        try:
            with patch("cloudtrain.config.manager.logger") as mock_logger:
                result = manager._load_config_file(config_file)

            assert result is None
            mock_logger.warning.assert_called()
        finally:
            config_file.unlink()

    def test_load_config_file_read_error(self):
        """Test handling read error when loading config file."""
        manager = ConfigManager.__new__(ConfigManager)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            config_file = Path(f.name)

        try:
            with patch("cloudtrain.config.manager.logger") as mock_logger:
                result = manager._load_config_file(config_file)

            assert result is None
            mock_logger.error.assert_called()
        finally:
            config_file.unlink()

    def test_configure_logging(self):
        """Test logging configuration."""
        settings = CloudTrainSettings(log_level=LogLevel.DEBUG)
        manager = ConfigManager.__new__(ConfigManager)
        manager.settings = settings

        with patch("cloudtrain.config.manager.logging") as mock_logging:
            mock_logger = Mock()
            mock_logging.getLogger.return_value = mock_logger
            mock_logger.handlers = []  # No existing handlers

            manager._configure_logging()

            mock_logging.getLogger.assert_called_with("cloudtrain")
            mock_logger.setLevel.assert_called_once()
            mock_logger.addHandler.assert_called_once()

    def test_sanitize_config(self):
        """Test configuration sanitization."""
        manager = ConfigManager.__new__(ConfigManager)
        config_dict = {
            "aws": {
                "access_key_id": "secret_key",
                "secret_access_key": "secret_value",
                "session_token": "secret_token",
            },
            "azure": {"client_secret": "secret_client", "region": "eastus"},
            "gcp": {
                "credentials_json": "secret_json",
                "service_account_key": "/path/to/key",
            },
            "safe_field": "safe_value",
        }

        result = manager._sanitize_config(config_dict)

        # Check that sensitive fields are masked
        assert result["aws"]["access_key_id"] == "***REDACTED***"
        assert result["aws"]["secret_access_key"] == "***REDACTED***"
        assert result["aws"]["session_token"] == "***REDACTED***"
        assert result["azure"]["client_secret"] == "***REDACTED***"
        assert result["gcp"]["credentials_json"] == "***REDACTED***"

        # Check that safe fields are preserved
        assert result["safe_field"] == "safe_value"
        assert result["azure"]["region"] == "eastus"

    def test_get_enabled_providers(self):
        """Test getting enabled providers."""
        settings = CloudTrainSettings(
            mock=MockConfig(enabled=True), aws=AWSConfig(enabled=False)
        )
        manager = ConfigManager(settings=settings)

        enabled = manager.get_enabled_providers()

        assert CloudProvider.MOCK in enabled
        assert CloudProvider.AWS not in enabled

    def test_get_provider_config_existing(self):
        """Test getting existing provider configuration."""
        mock_config = MockConfig(enabled=True)
        settings = CloudTrainSettings(mock=mock_config)
        manager = ConfigManager(settings=settings)

        result = manager.get_provider_config(CloudProvider.MOCK)

        assert result == mock_config

    def test_get_provider_config_default(self):
        """Test getting default provider configuration."""
        settings = CloudTrainSettings()
        manager = ConfigManager(settings=settings)

        result = manager.get_provider_config(CloudProvider.AWS)

        assert result is not None
        assert isinstance(result, AWSConfig)

    def test_is_provider_configured_true(self):
        """Test checking if provider is configured (true case)."""
        settings = CloudTrainSettings(mock=MockConfig(enabled=True))
        manager = ConfigManager(settings=settings)

        result = manager.is_provider_configured(CloudProvider.MOCK)

        assert result is True

    def test_is_provider_configured_false(self):
        """Test checking if provider is configured (false case)."""
        settings = CloudTrainSettings(mock=MockConfig(enabled=False))
        manager = ConfigManager(settings=settings)

        result = manager.is_provider_configured(CloudProvider.MOCK)

        assert result is False

    def test_is_provider_configured_missing(self):
        """Test checking if provider is configured (missing provider)."""
        settings = CloudTrainSettings()
        manager = ConfigManager(settings=settings)

        result = manager.is_provider_configured(CloudProvider.AWS)

        assert result is False

    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        settings = CloudTrainSettings(
            mock=MockConfig(enabled=True),
            aws=AWSConfig(enabled=False),
            azure=AzureConfig(enabled=False),
            gcp=GCPConfig(enabled=False),
        )
        manager = ConfigManager(settings=settings)

        result = manager.validate_configuration()

        assert result["valid"] is True
        assert "providers" in result
        assert "errors" in result
        assert result["providers"]["mock"]["valid"] is True

    def test_validate_configuration_invalid(self):
        """Test configuration validation with invalid config."""
        settings = CloudTrainSettings(
            aws=AWSConfig(enabled=True, region="")  # Invalid: empty region
        )
        manager = ConfigManager(settings=settings)

        result = manager.validate_configuration()

        assert result["valid"] is False
        assert "providers" in result
        assert "errors" in result
        assert result["providers"]["aws"]["valid"] is False
        assert len(result["providers"]["aws"]["errors"]) > 0

    def test_update_provider_config_success(self):
        """Test successful provider configuration update."""
        mock_config = MockConfig(enabled=True, failure_rate=0.1)
        settings = CloudTrainSettings(mock=mock_config)
        manager = ConfigManager(settings=settings)

        updates = {"failure_rate": 0.5}

        manager.update_provider_config(CloudProvider.MOCK, updates)

        updated_config = manager.get_provider_config(CloudProvider.MOCK)
        assert updated_config.failure_rate == 0.5

    def test_update_provider_config_not_configured(self):
        """Test updating provider configuration when not configured."""
        settings = CloudTrainSettings()
        manager = ConfigManager(settings=settings)

        # Mock get_provider_config to return None
        with patch.object(manager, "get_provider_config", return_value=None):
            updates = {"enabled": True}

            with pytest.raises(ConfigurationError, match="Provider mock not found"):
                manager.update_provider_config(CloudProvider.MOCK, updates)

    def test_update_provider_config_validation_error(self):
        """Test updating provider configuration with invalid data."""
        mock_config = MockConfig(enabled=True)
        settings = CloudTrainSettings(mock=mock_config)
        manager = ConfigManager(settings=settings)

        updates = {"failure_rate": 2.0}  # Invalid: > 1.0

        with pytest.raises(ConfigurationError, match="Invalid configuration update"):
            manager.update_provider_config(CloudProvider.MOCK, updates)

    def test_save_configuration_yaml(self):
        """Test saving configuration to YAML file."""
        settings = CloudTrainSettings(mock=MockConfig(enabled=True))
        manager = ConfigManager(settings=settings)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_file = Path(f.name)

        try:
            manager.save_configuration(output_file)

            # Verify file was created and contains expected content
            assert output_file.exists()
            with open(output_file, "r") as f:
                content = yaml.safe_load(f)
            assert "mock" in content
        finally:
            output_file.unlink()

    def test_save_configuration_json(self):
        """Test saving configuration to JSON file."""
        settings = CloudTrainSettings(mock=MockConfig(enabled=True))
        manager = ConfigManager(settings=settings)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = Path(f.name)

        try:
            manager.save_configuration(output_file)

            # Verify file was created and contains expected content
            assert output_file.exists()
            with open(output_file, "r") as f:
                content = json.load(f)
            assert "mock" in content
        finally:
            output_file.unlink()

    def test_save_configuration_unsupported_format(self):
        """Test saving configuration with unsupported format."""
        settings = CloudTrainSettings()
        manager = ConfigManager(settings=settings)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = Path(f.name)

        try:
            with pytest.raises(ConfigurationError, match="Unsupported output format"):
                manager.save_configuration(output_file)
        finally:
            if output_file.exists():
                output_file.unlink()

    def test_save_configuration_write_error(self):
        """Test handling write error when saving configuration."""
        settings = CloudTrainSettings()
        manager = ConfigManager(settings=settings)

        # Use a path that will cause a write error
        output_file = Path("/root/readonly/config.yaml")

        with pytest.raises(ConfigurationError, match="Failed to save configuration"):
            manager.save_configuration(output_file)

    def test_str_representation(self):
        """Test string representation of ConfigManager."""
        settings = CloudTrainSettings(
            mock=MockConfig(enabled=True), aws=AWSConfig(enabled=False)
        )
        manager = ConfigManager(settings=settings)

        str_repr = str(manager)

        assert "ConfigManager" in str_repr
        assert "mock" in str_repr
        assert "sources" in str_repr
        assert "providers" in str_repr
