"""Unit tests for CloudTrain configuration settings."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cloudtrain.config.settings import (
    AWSConfig,
    AzureConfig,
    BaseProviderConfig,
    CloudTrainSettings,
    GCPConfig,
    MockConfig,
)
from cloudtrain.enums import CloudProvider, LogLevel


class TestBaseProviderConfig:
    """Test BaseProviderConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BaseProviderConfig()

        assert config.enabled is True
        assert config.region is None
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BaseProviderConfig(
            enabled=False,
            region="us-west-2",
            timeout=600,
            max_retries=5,
            retry_delay=2.0,
        )

        assert config.enabled is False
        assert config.region == "us-west-2"
        assert config.timeout == 600
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_validation_timeout_bounds(self):
        """Test timeout validation bounds."""
        # Valid timeout
        config = BaseProviderConfig(timeout=300)
        assert config.timeout == 300

        # Invalid timeout - too low
        with pytest.raises(ValidationError):
            BaseProviderConfig(timeout=0)

        # Invalid timeout - too high
        with pytest.raises(ValidationError):
            BaseProviderConfig(timeout=4000)

    def test_validation_max_retries_bounds(self):
        """Test max_retries validation bounds."""
        # Valid max_retries
        config = BaseProviderConfig(max_retries=5)
        assert config.max_retries == 5

        # Invalid max_retries - too low
        with pytest.raises(ValidationError):
            BaseProviderConfig(max_retries=-1)

        # Invalid max_retries - too high
        with pytest.raises(ValidationError):
            BaseProviderConfig(max_retries=15)

    def test_validation_retry_delay_bounds(self):
        """Test retry_delay validation bounds."""
        # Valid retry_delay
        config = BaseProviderConfig(retry_delay=5.0)
        assert config.retry_delay == 5.0

        # Invalid retry_delay - too low
        with pytest.raises(ValidationError):
            BaseProviderConfig(retry_delay=0.05)

        # Invalid retry_delay - too high
        with pytest.raises(ValidationError):
            BaseProviderConfig(retry_delay=100.0)

    def test_is_valid_enabled_with_region(self):
        """Test is_valid returns True when enabled and has region."""
        config = BaseProviderConfig(enabled=True, region="us-east-1")
        assert config.is_valid() is True

    def test_is_valid_disabled(self):
        """Test is_valid returns False when disabled."""
        config = BaseProviderConfig(enabled=False, region="us-east-1")
        assert config.is_valid() is False

    def test_is_valid_no_region(self):
        """Test is_valid returns False when no region."""
        config = BaseProviderConfig(enabled=True, region=None)
        assert config.is_valid() is False

    def test_is_valid_empty_region(self):
        """Test is_valid returns False when region is empty string."""
        config = BaseProviderConfig(enabled=True, region="")
        assert config.is_valid() is False


class TestAWSConfig:
    """Test AWSConfig class."""

    def test_default_values(self):
        """Test default AWS configuration values."""
        config = AWSConfig()

        assert config.enabled is True
        assert config.region == "us-west-2"  # AWS default
        assert config.access_key_id is None
        assert config.secret_access_key is None
        assert config.session_token is None
        assert config.role_arn is None
        assert config.profile_name is None

    def test_custom_values(self):
        """Test custom AWS configuration values."""
        config = AWSConfig(
            region="eu-west-1",
            access_key_id="AKIATEST",
            secret_access_key="secret123",
            session_token="token123",
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            profile_name="test-profile",
        )

        assert config.region == "eu-west-1"
        assert config.access_key_id.get_secret_value() == "AKIATEST"
        assert config.secret_access_key.get_secret_value() == "secret123"
        assert config.session_token.get_secret_value() == "token123"
        assert config.role_arn == "arn:aws:iam::123456789012:role/TestRole"
        assert config.profile_name == "test-profile"

    def test_is_valid_with_access_keys(self):
        """Test is_valid returns True with access keys."""
        config = AWSConfig(
            region="us-east-1", access_key_id="AKIATEST", secret_access_key="secret123"
        )
        assert config.is_valid() is True

    def test_is_valid_with_profile(self):
        """Test is_valid returns True with profile."""
        config = AWSConfig(region="us-east-1", profile_name="default")
        assert config.is_valid() is True

    def test_is_valid_with_role(self):
        """Test is_valid returns True with role ARN."""
        config = AWSConfig(
            region="us-east-1", role_arn="arn:aws:iam::123456789012:role/TestRole"
        )
        assert config.is_valid() is True

    def test_is_valid_no_credentials(self):
        """Test is_valid returns False with no credentials."""
        config = AWSConfig(region="us-east-1")
        assert config.is_valid() is False

    def test_is_valid_partial_access_keys(self):
        """Test is_valid returns False with partial access keys."""
        config = AWSConfig(region="us-east-1", access_key_id="AKIATEST")
        assert config.is_valid() is False

    def test_is_valid_no_region(self):
        """Test is_valid returns False with no region."""
        # Can't set region=None due to validation, so test with empty string
        config = AWSConfig(
            region="", access_key_id="AKIATEST", secret_access_key="secret123"
        )
        assert config.is_valid() is False


class TestAzureConfig:
    """Test AzureConfig class."""

    def test_default_values(self):
        """Test default Azure configuration values."""
        config = AzureConfig()

        assert config.enabled is True
        assert config.region == "eastus"  # Azure default
        assert config.subscription_id is None
        assert config.resource_group is None
        assert config.workspace_name is None
        assert config.tenant_id is None
        assert config.client_id is None
        assert config.client_secret is None

    def test_custom_values(self):
        """Test custom Azure configuration values."""
        config = AzureConfig(
            region="westus2",
            subscription_id="12345678-1234-1234-1234-123456789012",
            resource_group="test-rg",
            workspace_name="test-workspace",
            tenant_id="87654321-4321-4321-4321-210987654321",
            client_id="11111111-1111-1111-1111-111111111111",
            client_secret="secret123",
        )

        assert config.region == "westus2"
        assert (
            config.subscription_id.get_secret_value()
            == "12345678-1234-1234-1234-123456789012"
        )
        assert config.resource_group == "test-rg"
        assert config.workspace_name == "test-workspace"
        assert (
            config.tenant_id.get_secret_value()
            == "87654321-4321-4321-4321-210987654321"
        )
        assert (
            config.client_id.get_secret_value()
            == "11111111-1111-1111-1111-111111111111"
        )
        assert config.client_secret.get_secret_value() == "secret123"

    def test_is_valid_with_service_principal(self):
        """Test is_valid returns True with service principal credentials."""
        config = AzureConfig(
            subscription_id="12345678-1234-1234-1234-123456789012",
            resource_group="test-rg",
            workspace_name="test-workspace",
            tenant_id="87654321-4321-4321-4321-210987654321",
            client_id="11111111-1111-1111-1111-111111111111",
            client_secret="secret123",
        )
        assert config.is_valid() is True

    def test_is_valid_missing_subscription(self):
        """Test is_valid returns False with missing subscription."""
        config = AzureConfig(
            tenant_id="87654321-4321-4321-4321-210987654321",
            client_id="11111111-1111-1111-1111-111111111111",
            client_secret="secret123",
        )
        assert config.is_valid() is False

    def test_is_valid_missing_tenant(self):
        """Test is_valid returns False with missing tenant."""
        config = AzureConfig(
            subscription_id="12345678-1234-1234-1234-123456789012",
            client_id="11111111-1111-1111-1111-111111111111",
            client_secret="secret123",
        )
        assert config.is_valid() is False

    def test_is_valid_missing_client_credentials(self):
        """Test is_valid returns False with missing client credentials."""
        config = AzureConfig(
            subscription_id="12345678-1234-1234-1234-123456789012",
            tenant_id="87654321-4321-4321-4321-210987654321",
        )
        assert config.is_valid() is False


class TestGCPConfig:
    """Test GCPConfig class."""

    def test_default_values(self):
        """Test default GCP configuration values."""
        config = GCPConfig()

        assert config.enabled is True
        assert config.region == "us-central1"  # GCP default
        assert config.project_id is None
        assert config.service_account_key is None
        assert config.credentials_json is None

    def test_custom_values(self):
        """Test custom GCP configuration values."""
        config = GCPConfig(
            region="us-west1",
            project_id="test-project-123",
            service_account_key=Path("/path/to/key.json"),
            credentials_json='{"type": "service_account"}',
        )

        assert config.region == "us-west1"
        assert config.project_id == "test-project-123"
        assert config.service_account_key == Path("/path/to/key.json")
        assert (
            config.credentials_json.get_secret_value() == '{"type": "service_account"}'
        )

    def test_is_valid_with_project_and_key_file(self):
        """Test is_valid returns True with project and key file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            key_path = Path(f.name)
            f.write(b'{"type": "service_account"}')

        try:
            config = GCPConfig(project_id="test-project", service_account_key=key_path)
            assert config.is_valid() is True
        finally:
            key_path.unlink()

    def test_is_valid_with_project_and_credentials_json(self):
        """Test is_valid returns True with project and credentials JSON."""
        config = GCPConfig(
            project_id="test-project", credentials_json='{"type": "service_account"}'
        )
        assert config.is_valid() is True

    def test_is_valid_missing_project(self):
        """Test is_valid returns False with missing project."""
        config = GCPConfig(credentials_json='{"type": "service_account"}')
        assert config.is_valid() is False

    def test_is_valid_missing_credentials(self):
        """Test is_valid returns False with missing credentials."""
        config = GCPConfig(project_id="test-project")
        assert config.is_valid() is False


class TestMockConfig:
    """Test MockConfig class."""

    def test_default_values(self):
        """Test default mock configuration values."""
        config = MockConfig()

        assert config.enabled is True
        assert config.simulate_failures is False
        assert config.failure_rate == 0.1
        assert config.response_delay == 0.1

    def test_custom_values(self):
        """Test custom mock configuration values."""
        config = MockConfig(
            enabled=False, simulate_failures=True, failure_rate=0.2, response_delay=0.5
        )

        assert config.enabled is False
        assert config.simulate_failures is True
        assert config.failure_rate == 0.2
        assert config.response_delay == 0.5

    def test_validation_failure_rate_bounds(self):
        """Test failure_rate validation bounds."""
        # Valid failure_rate
        config = MockConfig(failure_rate=0.5)
        assert config.failure_rate == 0.5

        # Invalid failure_rate - too low
        with pytest.raises(ValidationError):
            MockConfig(failure_rate=-0.1)

        # Invalid failure_rate - too high
        with pytest.raises(ValidationError):
            MockConfig(failure_rate=1.1)

    def test_validation_response_delay_bounds(self):
        """Test response_delay validation bounds."""
        # Valid response_delay
        config = MockConfig(response_delay=1.0)
        assert config.response_delay == 1.0

        # Invalid response_delay - too low
        with pytest.raises(ValidationError):
            MockConfig(response_delay=-0.1)

    def test_is_valid_enabled(self):
        """Test is_valid returns True when enabled."""
        config = MockConfig(enabled=True)
        assert config.is_valid() is True

    def test_is_valid_disabled(self):
        """Test is_valid returns False when disabled."""
        config = MockConfig(enabled=False)
        assert config.is_valid() is False


class TestCloudTrainSettings:
    """Test CloudTrainSettings class."""

    def test_default_values(self):
        """Test default CloudTrain settings values."""
        settings = CloudTrainSettings()

        assert settings.log_level == LogLevel.INFO
        assert settings.config_file is None
        assert settings.credentials_file is None
        assert settings.default_provider is None
        assert settings.auto_discover_providers is True
        assert settings.max_concurrent_jobs == 10
        assert settings.job_poll_interval == 30.0

        # Check provider defaults
        assert isinstance(settings.aws, AWSConfig)
        assert isinstance(settings.azure, AzureConfig)
        assert isinstance(settings.gcp, GCPConfig)
        assert isinstance(settings.mock, MockConfig)

    def test_custom_values(self):
        """Test custom CloudTrain settings values."""
        settings = CloudTrainSettings(
            log_level=LogLevel.DEBUG,
            default_provider=CloudProvider.AWS,
            auto_discover_providers=False,
            max_concurrent_jobs=5,
            job_poll_interval=10.0,
            aws=AWSConfig(region="eu-west-1"),
            mock=MockConfig(enabled=False),
        )

        assert settings.log_level == LogLevel.DEBUG
        assert settings.default_provider == CloudProvider.AWS
        assert settings.auto_discover_providers is False
        assert settings.max_concurrent_jobs == 5
        assert settings.job_poll_interval == 10.0
        assert settings.aws.region == "eu-west-1"
        assert settings.mock.enabled is False

    def test_validation_max_concurrent_jobs_bounds(self):
        """Test max_concurrent_jobs validation bounds."""
        # Valid max_concurrent_jobs
        settings = CloudTrainSettings(max_concurrent_jobs=5)
        assert settings.max_concurrent_jobs == 5

        # Invalid max_concurrent_jobs - too low
        with pytest.raises(ValidationError):
            CloudTrainSettings(max_concurrent_jobs=0)

        # Invalid max_concurrent_jobs - too high
        with pytest.raises(ValidationError):
            CloudTrainSettings(max_concurrent_jobs=101)

    def test_validation_job_poll_interval_bounds(self):
        """Test job_poll_interval validation bounds."""
        # Valid job_poll_interval
        settings = CloudTrainSettings(job_poll_interval=10.0)
        assert settings.job_poll_interval == 10.0

        # Invalid job_poll_interval - too low
        with pytest.raises(ValidationError):
            CloudTrainSettings(job_poll_interval=0.5)

        # Invalid job_poll_interval - too high
        with pytest.raises(ValidationError):
            CloudTrainSettings(job_poll_interval=301.0)

    def test_validate_file_paths_existing_file(self):
        """Test file path validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = Path(f.name)

        try:
            settings = CloudTrainSettings(config_file=file_path)
            assert settings.config_file == file_path
        finally:
            file_path.unlink()

    def test_validate_file_paths_nonexistent_file(self):
        """Test file path validation with non-existent file."""
        nonexistent_path = Path("/nonexistent/file.yaml")

        with pytest.raises(ValidationError, match="File does not exist"):
            CloudTrainSettings(config_file=nonexistent_path)

    def test_get_provider_config_existing(self):
        """Test getting existing provider configuration."""
        settings = CloudTrainSettings()

        aws_config = settings.get_provider_config(CloudProvider.AWS)
        assert isinstance(aws_config, AWSConfig)

        azure_config = settings.get_provider_config(CloudProvider.AZURE)
        assert isinstance(azure_config, AzureConfig)

        gcp_config = settings.get_provider_config(CloudProvider.GCP)
        assert isinstance(gcp_config, GCPConfig)

        mock_config = settings.get_provider_config(CloudProvider.MOCK)
        assert isinstance(mock_config, MockConfig)

    def test_get_provider_config_nonexistent(self):
        """Test getting non-existent provider configuration."""
        settings = CloudTrainSettings()

        # Create a mock provider enum value that doesn't exist in the mapping
        unknown_provider = type("MockProvider", (), {"value": "unknown"})()
        result = settings.get_provider_config(unknown_provider)
        assert result is None

    def test_get_enabled_providers_all_enabled(self):
        """Test getting enabled providers when all are enabled and valid."""
        settings = CloudTrainSettings(
            aws=AWSConfig(
                region="us-east-1", access_key_id="test", secret_access_key="test"
            ),
            azure=AzureConfig(
                subscription_id="12345678-1234-1234-1234-123456789012",
                resource_group="test-rg",
                workspace_name="test-workspace",
                tenant_id="87654321-4321-4321-4321-210987654321",
                client_id="11111111-1111-1111-1111-111111111111",
                client_secret="secret123",
            ),
            gcp=GCPConfig(
                project_id="test-project",
                credentials_json='{"type": "service_account"}',
            ),
            mock=MockConfig(enabled=True),
        )

        enabled = settings.get_enabled_providers()

        assert CloudProvider.AWS in enabled
        assert CloudProvider.AZURE in enabled
        assert CloudProvider.GCP in enabled
        assert CloudProvider.MOCK in enabled

    def test_get_enabled_providers_some_disabled(self):
        """Test getting enabled providers when some are disabled."""
        settings = CloudTrainSettings(
            aws=AWSConfig(enabled=False),
            azure=AzureConfig(enabled=False),
            gcp=GCPConfig(enabled=False),
            mock=MockConfig(enabled=True),
        )

        enabled = settings.get_enabled_providers()

        assert CloudProvider.AWS not in enabled
        assert CloudProvider.AZURE not in enabled
        assert CloudProvider.GCP not in enabled
        assert CloudProvider.MOCK in enabled

    def test_get_enabled_providers_invalid_configs(self):
        """Test getting enabled providers when configs are invalid."""
        settings = CloudTrainSettings(
            aws=AWSConfig(region="us-east-1"),  # No credentials - invalid
            azure=AzureConfig(),  # No credentials - invalid
            gcp=GCPConfig(),  # No project/credentials - invalid
            mock=MockConfig(enabled=True),  # Valid
        )

        enabled = settings.get_enabled_providers()

        assert CloudProvider.AWS not in enabled
        assert CloudProvider.AZURE not in enabled
        assert CloudProvider.GCP not in enabled
        assert CloudProvider.MOCK in enabled

    @patch.dict(os.environ, {"CLOUDTRAIN_LOG_LEVEL": "debug"})
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        settings = CloudTrainSettings()
        assert settings.log_level == LogLevel.DEBUG

    @patch.dict(
        os.environ, {"AWS_REGION": "eu-central-1", "AWS_ACCESS_KEY_ID": "test-key"}
    )
    def test_provider_environment_variable_loading(self):
        """Test loading provider configuration from environment variables."""
        settings = CloudTrainSettings()
        assert settings.aws.region == "eu-central-1"
        assert settings.aws.access_key_id.get_secret_value() == "test-key"
