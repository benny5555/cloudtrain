"""Unit tests for base cloud provider classes."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.providers.base import (
    AuthenticationError,
    BaseCloudProvider,
    JobSubmissionError,
    ProviderError,
    ResourceNotAvailableError,
)
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)


class TestProviderError:
    """Test ProviderError exception class."""

    def test_provider_error_creation(self):
        """Test creating ProviderError with basic parameters."""
        error = ProviderError("Test error", CloudProvider.AWS)

        assert str(error) == "Test error"
        assert error.provider == CloudProvider.AWS
        assert error.error_code is None

    def test_provider_error_with_code(self):
        """Test creating ProviderError with error code."""
        error = ProviderError("Test error", CloudProvider.AZURE, "ERR001")

        assert str(error) == "Test error"
        assert error.provider == CloudProvider.AZURE
        assert error.error_code == "ERR001"


class TestAuthenticationError:
    """Test AuthenticationError exception class."""

    def test_authentication_error_creation(self):
        """Test creating AuthenticationError."""
        error = AuthenticationError("Auth failed", CloudProvider.GCP)

        assert str(error) == "Auth failed"
        assert error.provider == CloudProvider.GCP
        assert isinstance(error, ProviderError)


class TestResourceNotAvailableError:
    """Test ResourceNotAvailableError exception class."""

    def test_resource_not_available_error_creation(self):
        """Test creating ResourceNotAvailableError."""
        error = ResourceNotAvailableError("No resources", CloudProvider.AWS)

        assert str(error) == "No resources"
        assert error.provider == CloudProvider.AWS
        assert isinstance(error, ProviderError)


class TestJobSubmissionError:
    """Test JobSubmissionError exception class."""

    def test_job_submission_error_creation(self):
        """Test creating JobSubmissionError."""
        error = JobSubmissionError("Submit failed", CloudProvider.MOCK)

        assert str(error) == "Submit failed"
        assert error.provider == CloudProvider.MOCK
        assert isinstance(error, ProviderError)


class ConcreteProvider(BaseCloudProvider):
    """Concrete implementation of BaseCloudProvider for testing."""

    def _get_provider_type(self) -> CloudProvider:
        return CloudProvider.MOCK

    def _load_configuration(self) -> dict:
        return {"test_config": "value"}

    async def _authenticate(self) -> None:
        pass

    def _map_instance_type(
        self, instance_type: InstanceType, custom_type: str = None
    ) -> str:
        if custom_type:
            return custom_type
        return {
            InstanceType.CPU_SMALL: "mock.cpu_small",
            InstanceType.CPU_MEDIUM: "mock.cpu_medium",
            InstanceType.CPU_LARGE: "mock.cpu_large",
            InstanceType.GPU_SMALL: "mock.gpu_small",
            InstanceType.GPU_MEDIUM: "mock.gpu_medium",
            InstanceType.GPU_LARGE: "mock.gpu_large",
        }[instance_type]

    def _map_job_status(self, provider_status: str) -> JobStatus:
        return {
            "running": JobStatus.RUNNING,
            "completed": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED,
            "pending": JobStatus.PENDING,
        }.get(provider_status, JobStatus.UNKNOWN)

    async def _submit_job_impl(self, job_spec: TrainingJobSpec) -> TrainingJobResult:
        from datetime import datetime

        return TrainingJobResult(
            job_id="test-job-123",
            job_name=job_spec.job_name,
            provider=CloudProvider.MOCK,
            status=JobStatus.PENDING,
            submission_time=datetime.fromisoformat("2023-01-01T00:00:00+00:00"),
        )

    async def _get_job_status_impl(self, job_id: str) -> JobStatusUpdate:
        from datetime import datetime

        return JobStatusUpdate(
            job_id=job_id,
            status=JobStatus.RUNNING,
            updated_time=datetime.fromisoformat("2023-01-01T01:00:00+00:00"),
        )

    async def _cancel_job_impl(self, job_id: str) -> bool:
        return True

    async def _list_jobs_impl(self, status_filter=None, limit=100):
        from datetime import datetime

        return [
            JobStatusUpdate(
                job_id="job-1",
                status=JobStatus.RUNNING,
                updated_time=datetime.fromisoformat("2023-01-01T01:00:00+00:00"),
            )
        ]


# Shared fixtures for BaseCloudProvider tests
@pytest.fixture
def mock_config_manager():
    """Create a mock configuration manager."""
    config_manager = Mock()
    config_manager.get_provider_config.return_value = {"test": "config"}
    return config_manager


@pytest.fixture
def provider(mock_config_manager):
    """Create a concrete provider instance for testing."""
    return ConcreteProvider(mock_config_manager)


class TestBaseCloudProviderInitialization:
    """Test BaseCloudProvider initialization and basic properties."""

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.provider_type == CloudProvider.MOCK
        assert provider.is_authenticated is False
        assert provider._client is None
        assert provider._config == {"test_config": "value"}

    def test_str_representation(self, provider):
        """Test string representation of provider."""
        result = str(provider)
        assert result == "ConcreteProvider(mock)"

    def test_repr_representation(self, provider):
        """Test detailed string representation of provider."""
        result = repr(provider)
        assert "ConcreteProvider" in result
        assert "provider_type=mock" in result
        assert "is_authenticated=False" in result


class TestBaseCloudProviderAuthentication:
    """Test BaseCloudProvider authentication functionality."""

    @pytest.mark.asyncio
    async def test_ensure_authenticated_when_not_authenticated(self, provider):
        """Test ensure_authenticated when provider is not authenticated."""
        assert provider.is_authenticated is False

        with patch.object(
            provider, "_authenticate", new_callable=AsyncMock
        ) as mock_auth:
            await provider.ensure_authenticated()

            mock_auth.assert_called_once()
            assert provider.is_authenticated is True

    @pytest.mark.asyncio
    async def test_ensure_authenticated_when_already_authenticated(self, provider):
        """Test ensure_authenticated when provider is already authenticated."""
        provider.is_authenticated = True

        with patch.object(
            provider, "_authenticate", new_callable=AsyncMock
        ) as mock_auth:
            await provider.ensure_authenticated()

            mock_auth.assert_not_called()
            assert provider.is_authenticated is True

    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test provider cleanup."""
        provider.is_authenticated = True
        provider._client = Mock()

        await provider.close()

        assert provider.is_authenticated is False
        assert provider._client is None


class TestBaseCloudProviderValidation:
    """Test BaseCloudProvider job specification validation."""

    def test_validate_job_spec_valid(self, provider):
        """Test job spec validation with valid specification."""
        job_spec = TrainingJobSpec(
            job_name="test-job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_MEDIUM, instance_count=1
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["/data/input"], output_path="/data/output"
            ),
            environment_configuration=EnvironmentConfiguration(
                python_version="3.9",
                requirements_file="requirements.txt",
                entry_point="train.py",
            ),
        )

        # Should not raise any exception
        provider.validate_job_spec(job_spec)

    def test_validate_job_spec_with_custom_instance_type(self, provider):
        """Test job spec validation with custom instance type."""
        job_spec = TrainingJobSpec(
            job_name="test-job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CUSTOM,
                custom_instance_type="custom.large",
                instance_count=1,
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["/data/input"], output_path="/data/output"
            ),
            environment_configuration=EnvironmentConfiguration(
                python_version="3.9",
                requirements_file="requirements.txt",
                entry_point="train.py",
            ),
        )

        # Should not raise any exception
        provider.validate_job_spec(job_spec)

    def test_validate_job_spec_custom_without_type(self, provider):
        """Test job spec validation with custom instance but no custom type."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="custom_instance_type is required"):
            TrainingJobSpec(
                job_name="test-job",
                resource_requirements=ResourceRequirements(
                    instance_type=InstanceType.CUSTOM, instance_count=1
                ),
                data_configuration=DataConfiguration(
                    input_data_paths=["/data/input"], output_path="/data/output"
                ),
                environment_configuration=EnvironmentConfiguration(
                    python_version="3.9",
                    requirements_file="requirements.txt",
                    entry_point="train.py",
                ),
            )


class TestBaseCloudProviderMapping:
    """Test BaseCloudProvider instance type and status mapping."""

    @pytest.mark.parametrize(
        "instance_type,expected",
        [
            (InstanceType.CPU_SMALL, "mock.cpu_small"),
            (InstanceType.CPU_MEDIUM, "mock.cpu_medium"),
            (InstanceType.CPU_LARGE, "mock.cpu_large"),
            (InstanceType.GPU_SMALL, "mock.gpu_small"),
            (InstanceType.GPU_MEDIUM, "mock.gpu_medium"),
            (InstanceType.GPU_LARGE, "mock.gpu_large"),
        ],
    )
    def test_map_instance_type_standard(self, provider, instance_type, expected):
        """Test mapping standard instance types."""
        assert provider._map_instance_type(instance_type) == expected

    def test_map_instance_type_custom(self, provider):
        """Test mapping custom instance type."""
        result = provider._map_instance_type(InstanceType.CUSTOM, "custom.huge")
        assert result == "custom.huge"

    @pytest.mark.parametrize(
        "provider_status,expected",
        [
            ("running", JobStatus.RUNNING),
            ("completed", JobStatus.COMPLETED),
            ("failed", JobStatus.FAILED),
            ("pending", JobStatus.PENDING),
            ("unknown_status", JobStatus.UNKNOWN),
        ],
    )
    def test_map_job_status(self, provider, provider_status, expected):
        """Test mapping provider-specific job statuses."""
        assert provider._map_job_status(provider_status) == expected


class TestBaseCloudProviderJobOperations:
    """Test BaseCloudProvider job operations (submit, status, cancel, list)."""

    @pytest.mark.asyncio
    async def test_submit_job_success(self, provider):
        """Test successful job submission."""
        job_spec = TrainingJobSpec(
            job_name="test-job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_MEDIUM, instance_count=1
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["/data/input"], output_path="/data/output"
            ),
            environment_configuration=EnvironmentConfiguration(
                python_version="3.9",
                requirements_file="requirements.txt",
                entry_point="train.py",
            ),
        )

        with patch.object(provider, "ensure_authenticated", new_callable=AsyncMock):
            result = await provider.submit_job(job_spec)

            assert result.job_id == "test-job-123"
            assert result.status == JobStatus.PENDING
            assert result.provider == CloudProvider.MOCK

    @pytest.mark.asyncio
    async def test_submit_job_validation_error(self, provider):
        """Test job submission with validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="custom_instance_type is required"):
            TrainingJobSpec(
                job_name="test-job",
                resource_requirements=ResourceRequirements(
                    instance_type=InstanceType.CUSTOM,  # Missing custom_instance_type
                    instance_count=1,
                ),
                data_configuration=DataConfiguration(
                    input_data_paths=["/data/input"], output_path="/data/output"
                ),
                environment_configuration=EnvironmentConfiguration(
                    python_version="3.9", requirements_file="requirements.txt"
                ),
            )

    @pytest.mark.asyncio
    async def test_submit_job_implementation_error(self, provider):
        """Test job submission with implementation error."""
        job_spec = TrainingJobSpec(
            job_name="test-job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_MEDIUM, instance_count=1
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["/data/input"], output_path="/data/output"
            ),
            environment_configuration=EnvironmentConfiguration(
                python_version="3.9",
                requirements_file="requirements.txt",
                entry_point="train.py",
            ),
        )

        with (
            patch.object(provider, "ensure_authenticated", new_callable=AsyncMock),
            patch.object(
                provider, "_submit_job_impl", side_effect=Exception("Submit failed")
            ),
        ):

            with pytest.raises(JobSubmissionError, match="Job submission failed"):
                await provider.submit_job(job_spec)

    @pytest.mark.asyncio
    async def test_get_job_status_success(self, provider):
        """Test successful job status retrieval."""
        with patch.object(provider, "ensure_authenticated", new_callable=AsyncMock):
            result = await provider.get_job_status("test-job-123")

            assert result.job_id == "test-job-123"
            assert result.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_job_status_error(self, provider):
        """Test job status retrieval with error."""
        with (
            patch.object(provider, "ensure_authenticated", new_callable=AsyncMock),
            patch.object(
                provider, "_get_job_status_impl", side_effect=Exception("Status failed")
            ),
        ):

            with pytest.raises(ProviderError, match="Status retrieval failed"):
                await provider.get_job_status("test-job-123")

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, provider):
        """Test successful job cancellation."""
        with patch.object(provider, "ensure_authenticated", new_callable=AsyncMock):
            result = await provider.cancel_job("test-job-123")

            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_job_error(self, provider):
        """Test job cancellation with error."""
        with (
            patch.object(provider, "ensure_authenticated", new_callable=AsyncMock),
            patch.object(
                provider, "_cancel_job_impl", side_effect=Exception("Cancel failed")
            ),
        ):

            with pytest.raises(ProviderError, match="Job cancellation failed"):
                await provider.cancel_job("test-job-123")

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, provider):
        """Test successful job listing."""
        with patch.object(provider, "ensure_authenticated", new_callable=AsyncMock):
            result = await provider.list_jobs()

            assert len(result) == 1
            assert result[0].job_id == "job-1"
            assert result[0].status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_list_jobs_with_filter(self, provider):
        """Test job listing with status filter."""
        with patch.object(provider, "ensure_authenticated", new_callable=AsyncMock):
            result = await provider.list_jobs(status_filter=JobStatus.RUNNING, limit=50)

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_error(self, provider):
        """Test job listing with error."""
        with (
            patch.object(provider, "ensure_authenticated", new_callable=AsyncMock),
            patch.object(
                provider, "_list_jobs_impl", side_effect=Exception("List failed")
            ),
        ):

            with pytest.raises(ProviderError, match="Job listing failed"):
                await provider.list_jobs()
