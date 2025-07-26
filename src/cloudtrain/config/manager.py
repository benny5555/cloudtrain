"""Configuration manager for CloudTrain.

This module provides the ConfigManager class that handles loading,
validation, and management of configuration settings and credentials
for the CloudTrain universal cloud training API.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from cloudtrain.config.settings import BaseProviderConfig, CloudTrainSettings
from cloudtrain.enums import CloudProvider

logger: logging.Logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class ConfigManager:
    """Configuration manager for CloudTrain.

    This class handles loading configuration from various sources including
    environment variables, configuration files, and programmatic settings.
    It provides a unified interface for accessing provider-specific
    configurations and credentials.

    Attributes:
        settings: Main CloudTrain settings
        config_sources: List of configuration sources that were loaded
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        settings: Optional[CloudTrainSettings] = None,
    ) -> None:
        """Initialize the configuration manager.

        Args:
            config_file: Optional path to configuration file
            settings: Optional pre-configured settings
        """
        self.config_sources: list[str] = []

        if settings:
            self.settings = settings
            self.config_sources.append("programmatic")
        else:
            self.settings = self._load_settings(config_file)

        # Set up logging based on configuration
        self._configure_logging()

        logger.info(f"ConfigManager initialized with sources: {self.config_sources}")

    def _load_settings(
        self, config_file: Optional[Union[str, Path]] = None
    ) -> CloudTrainSettings:
        """Load settings from various sources.

        Args:
            config_file: Optional path to configuration file

        Returns:
            Loaded CloudTrain settings

        Raises:
            ConfigurationError: If configuration loading fails
        """
        # Start with default settings (will load from environment)
        try:
            settings: CloudTrainSettings = CloudTrainSettings()
            self.config_sources.append("environment")
        except ValidationError as e:
            raise ConfigurationError(f"Invalid environment configuration: {e}")

        # Override with config file if provided
        if config_file:
            file_config: Optional[Dict[str, Any]] = self._load_config_file(config_file)
            if file_config:
                try:
                    # Merge file config with environment config
                    settings = CloudTrainSettings(**file_config)
                    self.config_sources.append(f"file:{config_file}")
                except ValidationError as e:
                    raise ConfigurationError(f"Invalid file configuration: {e}")

        # Look for default config files
        else:
            default_files: List[Path] = [
                Path.cwd() / "cloudtrain.yaml",
                Path.cwd() / "cloudtrain.yml",
                Path.cwd() / "cloudtrain.json",
                Path.home() / ".cloudtrain" / "config.yaml",
                Path.home() / ".cloudtrain" / "config.yml",
                Path.home() / ".cloudtrain" / "config.json",
            ]

            for config_path in default_files:
                if config_path.exists():
                    file_config = self._load_config_file(config_path)
                    if file_config:
                        try:
                            # Update settings with file config
                            for key, value in file_config.items():
                                if hasattr(settings, key):
                                    setattr(settings, key, value)
                            self.config_sources.append(f"file:{config_path}")
                            break
                        except Exception as e:
                            logger.warning(
                                f"Failed to load config from {config_path}: {e}"
                            )

        return settings

    def _load_config_file(
        self, config_file: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Load configuration from a file.

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration dictionary or None if loading fails
        """
        config_path: Path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    logger.warning(
                        f"Unsupported config file format: {config_path.suffix}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Failed to load configuration file {config_path}: {e}")
            return None

    def _configure_logging(self) -> None:
        """Configure logging based on settings."""
        log_level: str = self.settings.log_level.value.upper()

        # Configure root logger for CloudTrain
        cloudtrain_logger: logging.Logger = logging.getLogger("cloudtrain")
        cloudtrain_logger.setLevel(getattr(logging, log_level))

        # Add console handler if not already present
        if not cloudtrain_logger.handlers:
            handler: logging.StreamHandler = logging.StreamHandler()
            formatter: logging.Formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            cloudtrain_logger.addHandler(handler)

    def get_provider_config(
        self, provider: CloudProvider
    ) -> Optional[BaseProviderConfig]:
        """Get configuration for a specific provider.

        Args:
            provider: Cloud provider to get configuration for

        Returns:
            Provider configuration or None if not available
        """
        return self.settings.get_provider_config(provider)

    def is_provider_configured(self, provider: CloudProvider) -> bool:
        """Check if a provider is properly configured.

        Args:
            provider: Cloud provider to check

        Returns:
            True if provider is configured and enabled
        """
        config: Optional[BaseProviderConfig] = self.get_provider_config(provider)
        return config is not None and config.enabled and config.is_valid()

    def get_enabled_providers(self) -> list[CloudProvider]:
        """Get list of enabled and configured providers.

        Returns:
            List of available cloud providers
        """
        return self.settings.get_enabled_providers()

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration.

        Returns:
            Dictionary with validation results for each provider
        """
        results = {"valid": True, "providers": {}, "errors": []}

        for provider in CloudProvider:
            config = self.get_provider_config(provider)
            if config:
                provider_result = {
                    "enabled": config.enabled,
                    "valid": config.is_valid(),
                    "errors": [],
                }

                if config.enabled and not config.is_valid():
                    provider_result["errors"].append(
                        "Configuration is incomplete or invalid"
                    )
                    results["valid"] = False

                results["providers"][provider.value] = provider_result

        # Check if at least one provider is available
        if not any(self.is_provider_configured(p) for p in CloudProvider):
            results["valid"] = False
            results["errors"].append("No providers are properly configured")

        return results

    def update_provider_config(
        self, provider: CloudProvider, config_updates: Dict[str, Any]
    ) -> None:
        """Update configuration for a specific provider.

        Args:
            provider: Cloud provider to update
            config_updates: Dictionary of configuration updates

        Raises:
            ConfigurationError: If update fails
        """
        current_config = self.get_provider_config(provider)
        if not current_config:
            raise ConfigurationError(f"Provider {provider.value} not found")

        try:
            # Create updated configuration
            config_dict = current_config.model_dump()
            config_dict.update(config_updates)

            # Validate updated configuration
            config_class = type(current_config)
            updated_config = config_class(**config_dict)

            # Update settings
            setattr(self.settings, provider.value, updated_config)

            logger.info(f"Updated configuration for provider {provider.value}")

        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration update: {e}")

    def save_configuration(self, output_file: Union[str, Path]) -> None:
        """Save current configuration to a file.

        Args:
            output_file: Path to output configuration file

        Raises:
            ConfigurationError: If saving fails
        """
        output_path = Path(output_file)

        try:
            # Convert settings to dictionary
            config_dict = self.settings.model_dump(exclude_unset=True)

            # Remove sensitive information
            config_dict = self._sanitize_config(config_dict)

            # Save based on file extension
            with open(output_path, "w", encoding="utf-8") as f:
                if output_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif output_path.suffix.lower() == ".json":
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigurationError(
                        f"Unsupported output format: {output_path.suffix}"
                    )

            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def _sanitize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from configuration.

        Args:
            config_dict: Configuration dictionary to sanitize

        Returns:
            Sanitized configuration dictionary
        """
        # Fields to remove or mask
        sensitive_fields = [
            "access_key_id",
            "secret_access_key",
            "session_token",
            "client_secret",
            "credentials_json",
            "subscription_id",
            "tenant_id",
            "client_id",
        ]

        def sanitize_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    key: (
                        "***REDACTED***"
                        if key in sensitive_fields
                        else sanitize_recursive(value)
                    )
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize_recursive(item) for item in obj]
            else:
                return obj

        return sanitize_recursive(config_dict)

    def __str__(self) -> str:
        """Return string representation of the configuration manager."""
        enabled_providers = [p.value for p in self.get_enabled_providers()]
        return f"ConfigManager(sources={self.config_sources}, providers={enabled_providers})"
