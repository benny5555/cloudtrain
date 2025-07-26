"""Configuration settings for CloudTrain.

This module defines the configuration models and settings management
for the CloudTrain universal cloud training API.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings

from cloudtrain.enums import CloudProvider, LogLevel


class BaseProviderConfig(BaseSettings):
    """Base configuration for cloud providers.

    This class provides common configuration fields that are shared
    across different cloud provider implementations.

    Attributes:
        enabled: Whether this provider is enabled
        region: Default region for the provider
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
    """

    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    region: Optional[str] = Field(default=None, description="Default region")
    timeout: int = Field(
        default=300, ge=1, le=3600, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Base retry delay"
    )

    def is_valid(self) -> bool:
        """Check if the configuration is valid.

        Returns:
            True if the configuration is valid and complete
        """
        return self.enabled and bool(self.region)

    model_config = ConfigDict(env_prefix="", case_sensitive=False)


class AWSConfig(BaseProviderConfig):
    """Configuration for AWS SageMaker provider.

    Attributes:
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
        session_token: Optional AWS session token
        role_arn: Optional IAM role ARN to assume
        profile_name: Optional AWS profile name
        region: AWS region (required)
    """

    access_key_id: Optional[SecretStr] = Field(
        default=None, description="AWS access key ID"
    )
    secret_access_key: Optional[SecretStr] = Field(
        default=None, description="AWS secret access key"
    )
    session_token: Optional[SecretStr] = Field(
        default=None, description="AWS session token"
    )
    role_arn: Optional[str] = Field(default=None, description="IAM role ARN to assume")
    profile_name: Optional[str] = Field(default=None, description="AWS profile name")
    region: str = Field(default="us-west-2", description="AWS region")

    def is_valid(self) -> bool:
        """Check if AWS configuration is valid.

        Returns:
            True if configuration has valid credentials
        """
        if not self.enabled:
            return False

        # Check if we have credentials via access keys or profile
        has_access_keys = bool(self.access_key_id) and bool(self.secret_access_key)
        has_profile = bool(self.profile_name)
        has_role = bool(self.role_arn)

        return bool(self.region) and (has_access_keys or has_profile or has_role)

    model_config = ConfigDict(env_prefix="AWS_")


class AzureConfig(BaseProviderConfig):
    """Configuration for Azure Machine Learning provider.

    Attributes:
        subscription_id: Azure subscription ID
        resource_group: Azure resource group name
        workspace_name: Azure ML workspace name
        tenant_id: Azure tenant ID
        client_id: Azure client ID
        client_secret: Azure client secret
        region: Azure region
    """

    subscription_id: Optional[SecretStr] = Field(
        default=None, description="Azure subscription ID"
    )
    resource_group: Optional[str] = Field(
        default=None, description="Resource group name"
    )
    workspace_name: Optional[str] = Field(default=None, description="ML workspace name")
    tenant_id: Optional[SecretStr] = Field(default=None, description="Azure tenant ID")
    client_id: Optional[SecretStr] = Field(default=None, description="Azure client ID")
    client_secret: Optional[SecretStr] = Field(
        default=None, description="Azure client secret"
    )
    region: str = Field(default="eastus", description="Azure region")

    def is_valid(self) -> bool:
        """Check if Azure configuration is valid.

        Returns:
            True if configuration has valid credentials
        """
        if not self.enabled:
            return False

        required_fields = [
            self.subscription_id,
            self.resource_group,
            self.workspace_name,
            self.region,
        ]

        # Check if we have service principal credentials
        has_sp_creds = (
            bool(self.tenant_id) and bool(self.client_id) and bool(self.client_secret)
        )

        return all(field for field in required_fields) and has_sp_creds

    model_config = ConfigDict(env_prefix="AZURE_")


class GCPConfig(BaseProviderConfig):
    """Configuration for Google Cloud AI Platform provider.

    Attributes:
        project_id: GCP project ID
        service_account_key: Path to service account key file
        credentials_json: Service account credentials as JSON string
        region: GCP region
    """

    project_id: Optional[str] = Field(default=None, description="GCP project ID")
    service_account_key: Optional[Path] = Field(
        default=None, description="Service account key file"
    )
    credentials_json: Optional[SecretStr] = Field(
        default=None, description="Credentials JSON"
    )
    region: str = Field(default="us-central1", description="GCP region")

    def is_valid(self) -> bool:
        """Check if GCP configuration is valid.

        Returns:
            True if configuration has valid credentials
        """
        if not self.enabled:
            return False

        has_key_file = self.service_account_key and self.service_account_key.exists()
        has_json_creds = bool(self.credentials_json)

        return (
            bool(self.project_id)
            and bool(self.region)
            and (has_key_file or has_json_creds)
        )

    model_config = ConfigDict(env_prefix="GCP_")


class MockConfig(BaseProviderConfig):
    """Configuration for mock provider.

    Attributes:
        simulate_failures: Whether to simulate random failures
        failure_rate: Probability of simulated failures (0.0 to 1.0)
        response_delay: Simulated response delay in seconds
    """

    simulate_failures: bool = Field(
        default=False, description="Simulate random failures"
    )
    failure_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Failure probability"
    )
    response_delay: float = Field(
        default=0.1, ge=0.0, le=10.0, description="Response delay"
    )

    def is_valid(self) -> bool:
        """Check if mock configuration is valid.

        Returns:
            Always True for mock provider
        """
        return self.enabled

    model_config = ConfigDict(env_prefix="MOCK_")


class CloudTrainSettings(BaseSettings):
    """Main configuration settings for CloudTrain.

    This class manages the overall configuration for the CloudTrain API,
    including provider-specific settings, logging, and global options.

    Attributes:
        log_level: Global logging level
        config_file: Path to configuration file
        credentials_file: Path to credentials file
        default_provider: Default cloud provider to use
        auto_discover_providers: Whether to auto-discover providers
        max_concurrent_jobs: Maximum concurrent job submissions
        job_poll_interval: Interval for polling job status in seconds
        providers: Provider-specific configurations
    """

    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Global logging level"
    )
    config_file: Optional[Path] = Field(
        default=None, description="Configuration file path"
    )
    credentials_file: Optional[Path] = Field(
        default=None, description="Credentials file path"
    )
    default_provider: Optional[CloudProvider] = Field(
        default=None, description="Default provider"
    )
    auto_discover_providers: bool = Field(
        default=True, description="Auto-discover providers"
    )
    max_concurrent_jobs: int = Field(
        default=10, ge=1, le=100, description="Max concurrent jobs"
    )
    job_poll_interval: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Job poll interval"
    )

    # Provider configurations
    aws: AWSConfig = Field(default_factory=AWSConfig, description="AWS configuration")
    azure: AzureConfig = Field(
        default_factory=AzureConfig, description="Azure configuration"
    )
    gcp: GCPConfig = Field(default_factory=GCPConfig, description="GCP configuration")
    mock: MockConfig = Field(
        default_factory=MockConfig, description="Mock configuration"
    )

    @field_validator("config_file", "credentials_file")
    @classmethod
    def validate_file_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate file paths exist if specified.

        Args:
            v: File path to validate

        Returns:
            Validated file path

        Raises:
            ValueError: If file doesn't exist
        """
        if v is not None and not v.exists():
            raise ValueError(f"File does not exist: {v}")
        return v

    def get_provider_config(
        self, provider: CloudProvider
    ) -> Optional[BaseProviderConfig]:
        """Get configuration for a specific provider.

        Args:
            provider: Cloud provider to get configuration for

        Returns:
            Provider configuration or None if not found
        """
        provider_configs = {
            CloudProvider.AWS: self.aws,
            CloudProvider.AZURE: self.azure,
            CloudProvider.GCP: self.gcp,
            CloudProvider.MOCK: self.mock,
        }

        return provider_configs.get(provider)

    def get_enabled_providers(self) -> List[CloudProvider]:
        """Get list of enabled providers.

        Returns:
            List of enabled cloud providers
        """
        enabled = []

        for provider in CloudProvider:
            config = self.get_provider_config(provider)
            if config and config.enabled and config.is_valid():
                enabled.append(provider)

        return enabled

    model_config = ConfigDict(
        env_prefix="CLOUDTRAIN_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )
