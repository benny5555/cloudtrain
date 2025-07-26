"""Integration tests for MockProvider.

These tests verify the MockProvider works correctly with real CloudTrain API
interactions without mocking the provider itself.
"""

import asyncio
from datetime import datetime

import pytest

from cloudtrain.api import CloudTrainingAPI
from cloudtrain.config import CloudTrainSettings, ConfigManager
from cloudtrain.config.settings import MockConfig
from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    ResourceRequirements,
    TrainingJobSpec,
)


@pytest.fixture
def mock_config_manager():
    """Provide a config manager with mock provider enabled."""
    settings = CloudTrainSettings(
        mock=MockConfig(
            enabled=True,
            simulate_failures=False,
            failure_rate=0.0,
            response_delay=0.01,  # Fast for testing
        )
    )
    return ConfigManager(settings=settings)


@pytest.fixture
def sample_job_spec():
    """Provide a sample job specification for integration testing."""
    return TrainingJobSpec(
        job_name="integration-test-job",
        description="Integration test job for MockProvider",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL, instance_count=1, volume_size_gb=30
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["file:///tmp/test-data"],
            output_path="file:///tmp/test-output",
        ),
        environment_configuration=EnvironmentConfiguration(
            entry_point="train.py",
            framework="pytorch",
            framework_version="2.0.0",
            environment_variables={"TEST_MODE": "true"},
        ),
        tags={"test": "integration", "provider": "mock"},
    )


@pytest.mark.integration
class TestMockProviderIntegration:
    """Integration tests for MockProvider with CloudTrainingAPI."""

    @pytest.mark.asyncio
    async def test_full_job_lifecycle(self, mock_config_manager, sample_job_spec):
        """Test complete job lifecycle from submission to completion."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Verify mock provider is available
            providers = api.get_available_providers()
            assert CloudProvider.MOCK in providers

            # Submit job
            result = await api.submit_job(CloudProvider.MOCK, sample_job_spec)

            # Verify submission result
            assert result.job_id.startswith("mock-job-")
            assert result.job_name == sample_job_spec.job_name
            assert result.provider == CloudProvider.MOCK
            assert result.status == JobStatus.PENDING
            assert result.submission_time is not None
            assert result.estimated_cost is not None
            assert result.provider_job_url is not None

            job_id = result.job_id

            # Monitor job progress through different states
            # Mock provider: PENDING (5s) -> STARTING (15s) -> RUNNING (2min) -> COMPLETED
            max_iterations = 1300  # 130 seconds with 0.1s intervals
            iteration = 0
            seen_states = set()

            while iteration < max_iterations:
                status = await api.get_job_status(CloudProvider.MOCK, job_id)

                # Verify status structure
                assert status.job_id == job_id
                assert status.status in JobStatus
                assert status.updated_time is not None

                # Track states we've seen
                seen_states.add(status.status)

                # Check if job is complete
                if status.status.is_terminal():
                    # Should be completed (mock provider has 90% success rate with failure disabled)
                    assert status.status == JobStatus.COMPLETED

                    # Verify final status has expected fields
                    assert status.progress_percentage == 100.0
                    assert status.metrics is not None
                    assert len(status.logs) > 0

                    # Verify we saw the expected progression
                    assert JobStatus.PENDING in seen_states
                    break

                # For running jobs, verify progress fields
                if status.status == JobStatus.RUNNING:
                    assert status.progress_percentage is not None
                    assert 0 <= status.progress_percentage <= 100
                    assert status.metrics is not None

                iteration += 1
                await asyncio.sleep(0.1)  # Short delay

            assert (
                iteration < max_iterations
            ), f"Job did not complete within expected time. Last status: {status.status}, seen states: {seen_states}"

    @pytest.mark.asyncio
    async def test_job_cancellation(self, mock_config_manager, sample_job_spec):
        """Test job cancellation functionality."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Submit job
            result = await api.submit_job(CloudProvider.MOCK, sample_job_spec)
            job_id = result.job_id

            # Wait a moment for job to start
            await asyncio.sleep(0.1)

            # Cancel the job
            cancel_success = await api.cancel_job(CloudProvider.MOCK, job_id)
            assert cancel_success is True

            # Verify job is cancelled
            status = await api.get_job_status(CloudProvider.MOCK, job_id)
            assert status.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_job_listing(self, mock_config_manager, sample_job_spec):
        """Test job listing functionality."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Submit multiple jobs
            job_ids = []
            for i in range(3):
                spec = TrainingJobSpec(
                    job_name=f"list-test-job-{i}",
                    resource_requirements=sample_job_spec.resource_requirements,
                    data_configuration=sample_job_spec.data_configuration,
                    environment_configuration=sample_job_spec.environment_configuration,
                )
                result = await api.submit_job(CloudProvider.MOCK, spec)
                job_ids.append(result.job_id)

            # List all jobs
            all_jobs = await api.list_jobs(CloudProvider.MOCK, limit=10)

            # Verify we get our jobs back
            assert len(all_jobs) >= 3
            returned_job_ids = [job.job_id for job in all_jobs]

            for job_id in job_ids:
                assert job_id in returned_job_ids

            # Test filtering by status
            running_jobs = await api.list_jobs(
                CloudProvider.MOCK, status_filter=JobStatus.RUNNING, limit=5
            )

            # All returned jobs should have RUNNING status
            for job in running_jobs:
                assert job.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dry_run_validation(self, mock_config_manager, sample_job_spec):
        """Test dry run functionality."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Submit dry run
            result = await api.submit_job(
                CloudProvider.MOCK, sample_job_spec, dry_run=True
            )

            # Verify dry run result
            assert result.job_id == "dry-run-job-id"
            assert result.job_name == sample_job_spec.job_name
            assert result.provider == CloudProvider.MOCK
            assert result.status == JobStatus.PENDING

            # Verify no actual job was created
            all_jobs = await api.list_jobs(CloudProvider.MOCK)
            dry_run_jobs = [job for job in all_jobs if job.job_id == "dry-run-job-id"]
            assert len(dry_run_jobs) == 0

    @pytest.mark.parametrize(
        "custom_instance_type,expected_mapping",
        [
            ("mock.custom.xlarge", "mock.custom.xlarge"),
            ("custom.huge", "custom.huge"),
            ("ml.p4d.24xlarge", "ml.p4d.24xlarge"),
        ],
    )
    @pytest.mark.asyncio
    async def test_custom_instance_type(
        self, mock_config_manager, custom_instance_type, expected_mapping
    ):
        """Test custom instance type handling."""
        spec = TrainingJobSpec(
            job_name="custom-instance-test",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CUSTOM,
                custom_instance_type=custom_instance_type,
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["file:///tmp/data"], output_path="file:///tmp/output"
            ),
            environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
        )

        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Submit job with custom instance type
            result = await api.submit_job(CloudProvider.MOCK, spec)

            # Verify submission succeeded
            assert result.job_id.startswith("mock-job-")
            assert result.metadata["instance_type"] == expected_mapping

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config_manager):
        """Test error handling with invalid job specifications."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Test with invalid job name - should raise validation error during creation
            with pytest.raises(Exception):  # ValidationError from Pydantic
                invalid_spec = TrainingJobSpec(
                    job_name="",  # Invalid empty name
                    resource_requirements=ResourceRequirements(
                        instance_type=InstanceType.CPU_SMALL
                    ),
                    data_configuration=DataConfiguration(
                        input_data_paths=["file:///tmp/data"],
                        output_path="file:///tmp/output",
                    ),
                    environment_configuration=EnvironmentConfiguration(
                        entry_point="train.py"
                    ),
                )

    @pytest.mark.asyncio
    async def test_concurrent_job_submissions(
        self, mock_config_manager, sample_job_spec
    ):
        """Test concurrent job submissions."""
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:

            # Submit multiple jobs concurrently
            async def submit_job(job_index):
                spec = TrainingJobSpec(
                    job_name=f"concurrent-job-{job_index}",
                    resource_requirements=sample_job_spec.resource_requirements,
                    data_configuration=sample_job_spec.data_configuration,
                    environment_configuration=sample_job_spec.environment_configuration,
                )
                return await api.submit_job(CloudProvider.MOCK, spec)

            # Submit 5 jobs concurrently
            tasks = [submit_job(i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            # Verify all submissions succeeded
            assert len(results) == 5
            job_ids = [result.job_id for result in results]

            # All job IDs should be unique
            assert len(set(job_ids)) == 5

            # All should be mock jobs
            for result in results:
                assert result.job_id.startswith("mock-job-")
                assert result.provider == CloudProvider.MOCK


@pytest.mark.integration
@pytest.mark.slow
class TestMockProviderFailureSimulation:
    """Test MockProvider with failure simulation enabled."""

    @pytest.fixture
    def failure_config_manager(self):
        """Provide a config manager with failure simulation enabled."""
        settings = CloudTrainSettings(
            mock=MockConfig(
                enabled=True,
                simulate_failures=True,
                failure_rate=0.5,  # 50% failure rate
                response_delay=0.01,
            )
        )
        return ConfigManager(settings=settings)

    @pytest.mark.asyncio
    async def test_job_with_failure_simulation(
        self, failure_config_manager, sample_job_spec
    ):
        """Test job behavior with failure simulation enabled."""
        async with CloudTrainingAPI(config_manager=failure_config_manager) as api:

            # Submit multiple jobs to test failure rate
            results = []
            for i in range(10):
                spec = TrainingJobSpec(
                    job_name=f"failure-test-job-{i}",
                    resource_requirements=sample_job_spec.resource_requirements,
                    data_configuration=sample_job_spec.data_configuration,
                    environment_configuration=sample_job_spec.environment_configuration,
                )
                result = await api.submit_job(CloudProvider.MOCK, spec)
                results.append(result)

            # Wait for jobs to complete
            await asyncio.sleep(3.0)  # Give time for jobs to finish

            # Check final statuses
            final_statuses = []
            for result in results:
                status = await api.get_job_status(CloudProvider.MOCK, result.job_id)
                final_statuses.append(status.status)

            # Should have a mix of completed and failed jobs
            completed_count = sum(
                1 for status in final_statuses if status == JobStatus.COMPLETED
            )
            failed_count = sum(
                1 for status in final_statuses if status == JobStatus.FAILED
            )

            # With 50% failure rate, we should see some failures
            assert (
                failed_count > 0
            ), "Expected some jobs to fail with failure simulation enabled"
            assert (
                completed_count > 0
            ), "Expected some jobs to complete even with failure simulation"
