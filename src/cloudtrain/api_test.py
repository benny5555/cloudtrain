"""Unit tests for CloudTrain main API."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cloudtrain.api import CloudTrainingAPI
from cloudtrain.config import ConfigManager
from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager."""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_provider_config.return_value = Mock(
        is_valid=Mock(return_value=True)
    )
    return config_manager


@pytest.fixture
def sample_job_spec():
    """Provide a sample job specification for testing."""
    return TrainingJobSpec(
        job_name="test-job",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["s3://bucket/data"], output_path="s3://bucket/output"
        ),
        environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
    )


@pytest.fixture
def sample_job_result():
    """Provide a sample job result for testing."""
    return TrainingJobResult(
        job_id="test-job-123",
        job_name="test-job",
        provider=CloudProvider.MOCK,
        status=JobStatus.PENDING,
        submission_time=datetime.now(UTC),
    )


class TestCloudTrainingAPI:
    """Test CloudTrainingAPI class."""

    def test_init_with_config_manager(self, mock_config_manager):
        """Test API initialization with config manager."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        assert api.config_manager == mock_config_manager
        assert api.providers == {}

    def test_init_without_config_manager(self):
        """Test API initialization without config manager."""
        with patch("cloudtrain.api.ConfigManager") as mock_config_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            api = CloudTrainingAPI(auto_discover_providers=False)

            assert api.config_manager == mock_config_instance
            mock_config_class.assert_called_once()

    @patch("cloudtrain.api.CloudTrainingAPI._discover_providers")
    def test_init_with_auto_discover(self, mock_discover, mock_config_manager):
        """Test API initialization with auto-discovery enabled."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=True
        )

        mock_discover.assert_called_once()

    def test_register_provider(self, mock_config_manager):
        """Test manual provider registration."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )
        mock_provider = Mock()

        api.register_provider(CloudProvider.MOCK, mock_provider)

        assert CloudProvider.MOCK in api.providers
        assert api.providers[CloudProvider.MOCK] == mock_provider

    def test_get_available_providers(self, mock_config_manager):
        """Test getting available providers."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )
        mock_provider = Mock()
        api.register_provider(CloudProvider.MOCK, mock_provider)

        providers = api.get_available_providers()

        assert providers == [CloudProvider.MOCK]

    def test_get_available_providers_empty(self, mock_config_manager):
        """Test getting available providers when none are registered."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        providers = api.get_available_providers()

        assert providers == []

    # Job submission tests

    @pytest.mark.asyncio
    async def test_submit_job_success(
        self, mock_config_manager, sample_job_spec, sample_job_result
    ):
        """Test successful job submission."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.submit_job.return_value = sample_job_result
        api.register_provider(CloudProvider.MOCK, mock_provider)

        # Mock validation
        with patch("cloudtrain.api.validate_job_spec") as mock_validate:
            with patch("cloudtrain.api.retry_with_backoff") as mock_retry:
                mock_retry.return_value = sample_job_result

                result = await api.submit_job(CloudProvider.MOCK, sample_job_spec)

                assert result == sample_job_result
                mock_validate.assert_called_once_with(sample_job_spec)
                mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_job_dry_run(self, mock_config_manager, sample_job_spec):
        """Test job submission with dry run."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        api.register_provider(CloudProvider.MOCK, mock_provider)

        with patch("cloudtrain.api.validate_job_spec") as mock_validate:
            result = await api.submit_job(
                CloudProvider.MOCK, sample_job_spec, dry_run=True
            )

            assert result.job_id == "dry-run-job-id"
            assert result.job_name == sample_job_spec.job_name
            assert result.provider == CloudProvider.MOCK
            assert result.status == JobStatus.PENDING
            mock_validate.assert_called_once_with(sample_job_spec)
            mock_provider.submit_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_job_provider_not_available(
        self, mock_config_manager, sample_job_spec
    ):
        """Test job submission with unavailable provider."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        with pytest.raises(ValueError) as exc_info:
            await api.submit_job(CloudProvider.AWS, sample_job_spec)

        assert "Provider aws is not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_submit_job_validation_error(
        self, mock_config_manager, sample_job_spec
    ):
        """Test job submission with validation error."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        api.register_provider(CloudProvider.MOCK, mock_provider)

        with patch("cloudtrain.api.validate_job_spec") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid job spec")

            with pytest.raises(ValueError):
                await api.submit_job(CloudProvider.MOCK, sample_job_spec)

    @pytest.mark.asyncio
    async def test_submit_job_provider_error(
        self, mock_config_manager, sample_job_spec
    ):
        """Test job submission with provider error."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        api.register_provider(CloudProvider.MOCK, mock_provider)

        with patch("cloudtrain.api.validate_job_spec"):
            with patch("cloudtrain.api.retry_with_backoff") as mock_retry:
                mock_retry.side_effect = Exception("Provider error")

                with pytest.raises(RuntimeError) as exc_info:
                    await api.submit_job(CloudProvider.MOCK, sample_job_spec)

                assert "Job submission failed" in str(exc_info.value)

    # Job status operations tests

    @pytest.mark.asyncio
    async def test_get_job_status_success(self, mock_config_manager):
        """Test successful job status retrieval."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider and status
        mock_provider = AsyncMock()
        mock_status = JobStatusUpdate(
            job_id="test-job-123",
            status=JobStatus.RUNNING,
            updated_time=datetime.now(UTC),
        )
        mock_provider.get_job_status.return_value = mock_status
        api.register_provider(CloudProvider.MOCK, mock_provider)

        result = await api.get_job_status(CloudProvider.MOCK, "test-job-123")

        assert result == mock_status
        mock_provider.get_job_status.assert_called_once_with("test-job-123")

    @pytest.mark.asyncio
    async def test_get_job_status_provider_not_available(self, mock_config_manager):
        """Test job status retrieval with unavailable provider."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        with pytest.raises(ValueError) as exc_info:
            await api.get_job_status(CloudProvider.AWS, "test-job-123")

        assert "Provider aws is not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_job_status_provider_error(self, mock_config_manager):
        """Test job status retrieval with provider error."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.get_job_status.side_effect = Exception("Provider error")
        api.register_provider(CloudProvider.MOCK, mock_provider)

        with pytest.raises(RuntimeError) as exc_info:
            await api.get_job_status(CloudProvider.MOCK, "test-job-123")

        assert "Status retrieval failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, mock_config_manager):
        """Test successful job cancellation."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.cancel_job.return_value = True
        api.register_provider(CloudProvider.MOCK, mock_provider)

        result = await api.cancel_job(CloudProvider.MOCK, "test-job-123")

        assert result is True
        mock_provider.cancel_job.assert_called_once_with("test-job-123")

    @pytest.mark.asyncio
    async def test_cancel_job_failure(self, mock_config_manager):
        """Test job cancellation failure."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.cancel_job.return_value = False
        api.register_provider(CloudProvider.MOCK, mock_provider)

        result = await api.cancel_job(CloudProvider.MOCK, "test-job-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, mock_config_manager):
        """Test successful job listing."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider and jobs
        mock_provider = AsyncMock()
        mock_jobs = [
            JobStatusUpdate(
                job_id="job-1", status=JobStatus.RUNNING, updated_time=datetime.now(UTC)
            ),
            JobStatusUpdate(
                job_id="job-2",
                status=JobStatus.COMPLETED,
                updated_time=datetime.now(UTC),
            ),
        ]
        mock_provider.list_jobs.return_value = mock_jobs
        api.register_provider(CloudProvider.MOCK, mock_provider)

        result = await api.list_jobs(CloudProvider.MOCK, JobStatus.RUNNING, 10)

        assert result == mock_jobs
        mock_provider.list_jobs.assert_called_once_with(JobStatus.RUNNING, 10)

    @pytest.mark.asyncio
    async def test_close(self, mock_config_manager):
        """Test API cleanup."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Mock provider
        mock_provider = AsyncMock()
        api.register_provider(CloudProvider.MOCK, mock_provider)

        await api.close()

        mock_provider.close.assert_called_once()
        assert api.providers == {}

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config_manager):
        """Test API as async context manager."""
        mock_provider = AsyncMock()

        async with CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        ) as api:
            api.register_provider(CloudProvider.MOCK, mock_provider)
            assert CloudProvider.MOCK in api.providers

        # Should have called close
        mock_provider.close.assert_called_once()

    # Parameterized tests for provider error scenarios

    @pytest.mark.parametrize(
        "provider,expected_error_msg",
        [
            (CloudProvider.AWS, "Provider aws is not available"),
            (CloudProvider.AZURE, "Provider azure is not available"),
            (CloudProvider.GCP, "Provider gcp is not available"),
            (CloudProvider.ALIBABA, "Provider alibaba is not available"),
            (CloudProvider.TENCENT, "Provider tencent is not available"),
        ],
    )
    @pytest.mark.asyncio
    async def test_operations_with_unavailable_provider(
        self, mock_config_manager, sample_job_spec, provider, expected_error_msg
    ):
        """Test various operations with unavailable providers."""
        api = CloudTrainingAPI(
            config_manager=mock_config_manager, auto_discover_providers=False
        )

        # Test submit_job with unavailable provider
        with pytest.raises(ValueError) as exc_info:
            await api.submit_job(provider, sample_job_spec)
        assert expected_error_msg in str(exc_info.value)

        # Test get_job_status with unavailable provider
        with pytest.raises(ValueError) as exc_info:
            await api.get_job_status(provider, "test-job-123")
        assert expected_error_msg in str(exc_info.value)

        # Test cancel_job with unavailable provider
        with pytest.raises(ValueError) as exc_info:
            await api.cancel_job(provider, "test-job-123")
        assert expected_error_msg in str(exc_info.value)

        # Test list_jobs with unavailable provider
        with pytest.raises(ValueError) as exc_info:
            await api.list_jobs(provider)
        assert expected_error_msg in str(exc_info.value)
