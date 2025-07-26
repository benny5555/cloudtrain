"""Unit tests for mock cloud provider implementation."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.providers.mock.provider import MockProvider
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    ResourceRequirements,
    TrainingJobSpec,
)


class TestMockProvider:
    """Test MockProvider class."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock()
        config_manager.get_provider_config.return_value = {
            "enabled": True,
            "simulate_failures": False,
            "failure_rate": 0.0,
            "response_delay": 0.1,
        }
        return config_manager

    @pytest.fixture
    def provider(self, mock_config_manager):
        """Create a mock provider instance for testing."""
        return MockProvider(mock_config_manager)

    @pytest.fixture
    def sample_job_spec(self):
        """Create a sample job specification for testing."""
        return TrainingJobSpec(
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

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.provider_type == CloudProvider.MOCK
        assert provider.is_authenticated is False
        assert provider._client is None
        assert isinstance(provider.jobs, dict)
        assert isinstance(provider.job_progression, dict)
        assert len(provider.jobs) == 0
        assert len(provider.job_progression) == 0

    def test_get_provider_type(self, provider):
        """Test getting provider type."""
        assert provider._get_provider_type() == CloudProvider.MOCK

    def test_load_configuration_default(self, provider):
        """Test loading default configuration."""
        config = provider._load_configuration()

        assert config["region"] == "mock-region-1"
        assert config["endpoint"] == "https://mock.cloudtrain.ai"
        assert config["api_version"] == "v1"
        assert config["timeout"] == 30
        assert config["max_retries"] == 3

    def test_load_configuration_custom(self, mock_config_manager):
        """Test loading custom configuration."""
        # Create a mock config object with attributes
        mock_config = Mock()
        mock_config.simulate_failures = True
        mock_config.failure_rate = 0.2
        mock_config.response_delay = 0.5

        mock_config_manager.get_provider_config.return_value = mock_config

        provider = MockProvider(mock_config_manager)
        config = provider._load_configuration()

        assert config["simulate_failures"] is True
        assert config["failure_rate"] == 0.2
        assert config["response_delay"] == 0.5

    @pytest.mark.asyncio
    async def test_authenticate(self, provider):
        """Test authentication simulation."""
        start_time = datetime.now(UTC)
        await provider._authenticate()
        end_time = datetime.now(UTC)

        # Should take at least 0.1 seconds due to simulated delay
        assert (end_time - start_time).total_seconds() >= 0.1

    def test_map_instance_type_standard(self, provider):
        """Test mapping standard instance types."""
        assert provider._map_instance_type(InstanceType.CPU_SMALL) == "mock.cpu.small"
        assert provider._map_instance_type(InstanceType.CPU_MEDIUM) == "mock.cpu.medium"
        assert provider._map_instance_type(InstanceType.CPU_LARGE) == "mock.cpu.large"
        assert provider._map_instance_type(InstanceType.GPU_SMALL) == "mock.gpu.small"
        assert provider._map_instance_type(InstanceType.GPU_MEDIUM) == "mock.gpu.medium"
        assert provider._map_instance_type(InstanceType.GPU_LARGE) == "mock.gpu.large"

    def test_map_instance_type_custom(self, provider):
        """Test mapping custom instance type."""
        result = provider._map_instance_type(InstanceType.CUSTOM, "custom.huge")
        assert result == "custom.huge"

    def test_map_instance_type_unsupported(self, provider):
        """Test mapping unsupported instance type."""
        # Create a mock instance type that's not in the mapping
        with pytest.raises(ValueError, match="Unsupported instance type"):
            provider._map_instance_type(None)

    def test_map_job_status(self, provider):
        """Test mapping provider-specific job statuses."""
        assert provider._map_job_status("QUEUED") == JobStatus.PENDING
        assert provider._map_job_status("INITIALIZING") == JobStatus.STARTING
        assert provider._map_job_status("TRAINING") == JobStatus.RUNNING
        assert provider._map_job_status("COMPLETED") == JobStatus.COMPLETED
        assert provider._map_job_status("FAILED") == JobStatus.FAILED
        assert provider._map_job_status("CANCELLED") == JobStatus.STOPPED
        assert provider._map_job_status("STOPPING") == JobStatus.STOPPING
        assert provider._map_job_status("UNKNOWN_STATUS") == JobStatus.UNKNOWN

    def test_simulate_job_progression(self, provider):
        """Test job progression simulation setup."""
        job_id = "test-job-123"
        provider._simulate_job_progression(job_id)

        assert job_id in provider.job_progression
        progression = provider.job_progression[job_id]
        assert len(progression) == 4
        assert progression[0] == JobStatus.PENDING
        assert progression[1] == JobStatus.STARTING
        assert progression[2] == JobStatus.RUNNING
        assert progression[3] == JobStatus.COMPLETED

    def test_get_current_job_status_progression(self, provider):
        """Test getting current job status during progression."""
        job_id = "test-job-123"
        provider._simulate_job_progression(job_id)

        # Mock job submission time to control progression
        provider.jobs[job_id] = {
            "submission_time": datetime.now(UTC) - timedelta(seconds=5),
            "job_spec": Mock(),
        }

        status = provider._get_current_job_status(job_id)
        # Should be in progression (PENDING, STARTING, RUNNING, or COMPLETED)
        assert status in [
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
        ]

    def test_get_current_job_status_nonexistent(self, provider):
        """Test getting status for non-existent job."""
        status = provider._get_current_job_status("nonexistent-job")
        assert status == JobStatus.UNKNOWN

    def test_get_current_job_status_no_simulation(self, provider):
        """Test getting status when failure simulation is disabled."""
        job_id = "test-job-123"
        provider.jobs[job_id] = {
            "submission_time": datetime.now(UTC) - timedelta(seconds=100),  # Old job
            "job_spec": Mock(),
        }

        status = provider._get_current_job_status(job_id)
        assert status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_submit_job_success(self, provider, sample_job_spec):
        """Test successful job submission."""
        # No need to mock ensure_authenticated for _submit_job_impl
        result = await provider._submit_job_impl(sample_job_spec)

        assert result.job_name == "test-job"
        assert result.provider == CloudProvider.MOCK
        assert result.status == JobStatus.PENDING
        assert result.job_id in provider.jobs
        assert result.job_id in provider.job_progression

        # Check job is stored correctly
        stored_job = provider.jobs[result.job_id]
        assert stored_job["job_spec"] == sample_job_spec
        assert "submission_time" in stored_job

    @pytest.mark.asyncio
    async def test_submit_job_with_delay(self, mock_config_manager, sample_job_spec):
        """Test job submission with response delay."""
        mock_config_manager.get_provider_config.return_value = {
            "enabled": True,
            "simulate_failures": False,
            "failure_rate": 0.0,
            "response_delay": 0.2,
        }

        provider = MockProvider(mock_config_manager)

        start_time = datetime.now(UTC)
        result = await provider._submit_job_impl(sample_job_spec)
        end_time = datetime.now(UTC)

        # Should take at least 0.2 seconds due to response delay
        assert (end_time - start_time).total_seconds() >= 0.2
        assert result.job_name == "test-job"

    @pytest.mark.asyncio
    async def test_get_job_status_existing_job(self, provider, sample_job_spec):
        """Test getting status for existing job."""
        # Submit a job first
        result = await provider._submit_job_impl(sample_job_spec)
        job_id = result.job_id

        # Get status
        status_update = await provider._get_job_status_impl(job_id)

        assert status_update.job_id == job_id
        assert status_update.status in [
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
        ]
        assert status_update.updated_time is not None

    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent_job(self, provider):
        """Test getting status for non-existent job."""
        with pytest.raises(ValueError, match="Job nonexistent-job not found"):
            await provider._get_job_status_impl("nonexistent-job")

    @pytest.mark.asyncio
    async def test_cancel_job_existing_job(self, provider, sample_job_spec):
        """Test cancelling existing job."""
        # Submit a job first
        result = await provider._submit_job_impl(sample_job_spec)
        job_id = result.job_id

        # Cancel the job
        cancelled = await provider._cancel_job_impl(job_id)

        assert cancelled is True
        # Check job status is updated
        status_update = await provider._get_job_status_impl(job_id)
        assert status_update.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_cancel_job_nonexistent_job(self, provider):
        """Test cancelling non-existent job."""
        cancelled = await provider._cancel_job_impl("nonexistent-job")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_job_already_completed(self, provider, sample_job_spec):
        """Test cancelling already completed job."""
        # Submit a job and mark it as completed
        result = await provider._submit_job_impl(sample_job_spec)
        job_id = result.job_id

        # Simulate job completion by setting old submission time
        provider.jobs[job_id]["submission_time"] = datetime.now(UTC) - timedelta(
            seconds=100
        )

        # Try to cancel
        cancelled = await provider._cancel_job_impl(job_id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, provider):
        """Test listing jobs when no jobs exist."""
        jobs = await provider._list_jobs_impl()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_list_jobs_with_jobs(self, provider, sample_job_spec):
        """Test listing jobs when jobs exist."""
        # Submit multiple jobs
        result1 = await provider._submit_job_impl(sample_job_spec)

        sample_job_spec.job_name = "test-job-2"
        result2 = await provider._submit_job_impl(sample_job_spec)

        # List all jobs
        jobs = await provider._list_jobs_impl()

        assert len(jobs) == 2
        job_ids = [job.job_id for job in jobs]
        assert result1.job_id in job_ids
        assert result2.job_id in job_ids

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, provider, sample_job_spec):
        """Test listing jobs with status filter."""
        # Submit a job and cancel it
        result = await provider._submit_job_impl(sample_job_spec)
        await provider._cancel_job_impl(result.job_id)

        # Submit another job
        sample_job_spec.job_name = "test-job-2"
        await provider._submit_job_impl(sample_job_spec)

        # List only stopped jobs
        stopped_jobs = await provider._list_jobs_impl(status_filter=JobStatus.STOPPED)
        assert len(stopped_jobs) == 1
        assert stopped_jobs[0].job_id == result.job_id
        assert stopped_jobs[0].status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_list_jobs_with_limit(self, provider, sample_job_spec):
        """Test listing jobs with limit."""
        # Submit multiple jobs
        for i in range(5):
            sample_job_spec.job_name = f"test-job-{i}"
            await provider._submit_job_impl(sample_job_spec)

        # List with limit
        jobs = await provider._list_jobs_impl(limit=3)
        assert len(jobs) == 3

    def test_str_representation(self, provider):
        """Test string representation of provider."""
        result = str(provider)
        assert result == "MockProvider(mock)"

    def test_repr_representation(self, provider):
        """Test detailed string representation of provider."""
        result = repr(provider)
        assert "MockProvider" in result
        assert "provider_type=mock" in result
        assert "is_authenticated=False" in result
