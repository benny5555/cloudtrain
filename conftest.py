"""Pytest configuration and shared fixtures for CloudTrain tests.

This module provides common test fixtures and configuration that can be used
across all CloudTrain test modules.
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock

import pytest

from cloudtrain.config import CloudTrainSettings, ConfigManager
from cloudtrain.config.settings import AWSConfig, AzureConfig, GCPConfig, MockConfig
from cloudtrain.enums import CloudProvider, InstanceType, JobStatus, LogLevel
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager for testing."""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_provider_config.return_value = Mock(
        enabled=True, is_valid=Mock(return_value=True)
    )
    config_manager.get_enabled_providers.return_value = [CloudProvider.MOCK]
    config_manager.is_provider_configured.return_value = True
    return config_manager


@pytest.fixture
def test_config_manager():
    """Provide a real configuration manager with test settings."""
    settings = CloudTrainSettings(
        log_level=LogLevel.DEBUG,
        auto_discover_providers=False,
        mock=MockConfig(
            enabled=True, simulate_failures=False, failure_rate=0.0, response_delay=0.01
        ),
        aws=AWSConfig(enabled=False, region="us-west-2"),
        azure=AzureConfig(enabled=False, region="eastus"),
        gcp=GCPConfig(enabled=False, region="us-central1"),
    )
    return ConfigManager(settings=settings)


@pytest.fixture
def sample_resource_requirements():
    """Provide sample resource requirements for testing."""
    return ResourceRequirements(
        instance_type=InstanceType.CPU_SMALL,
        instance_count=1,
        volume_size_gb=30,
        max_runtime_seconds=3600,
    )


@pytest.fixture
def sample_data_configuration():
    """Provide sample data configuration for testing."""
    return DataConfiguration(
        input_data_paths=["s3://test-bucket/input-data/"],
        output_path="s3://test-bucket/output/",
        checkpoint_path="s3://test-bucket/checkpoints/",
        data_format="parquet",
    )


@pytest.fixture
def sample_environment_configuration():
    """Provide sample environment configuration for testing."""
    return EnvironmentConfiguration(
        entry_point="train.py",
        framework="pytorch",
        framework_version="2.0.0",
        python_version="3.9",
        environment_variables={"CUDA_VISIBLE_DEVICES": "0", "BATCH_SIZE": "32"},
        command_line_args=["--epochs", "10", "--lr", "0.001"],
    )


@pytest.fixture
def sample_job_spec(
    sample_resource_requirements,
    sample_data_configuration,
    sample_environment_configuration,
):
    """Provide a complete sample job specification for testing."""
    return TrainingJobSpec(
        job_name="test-training-job",
        description="Sample training job for testing",
        resource_requirements=sample_resource_requirements,
        data_configuration=sample_data_configuration,
        environment_configuration=sample_environment_configuration,
        tags={"project": "test-project", "team": "ml-team", "environment": "testing"},
        provider_specific_config={"test_mode": True, "debug": True},
    )


@pytest.fixture
def sample_job_result():
    """Provide a sample job result for testing."""
    return TrainingJobResult(
        job_id="test-job-12345",
        job_name="test-training-job",
        provider=CloudProvider.MOCK,
        status=JobStatus.PENDING,
        submission_time=datetime.utcnow(),
        estimated_start_time=datetime.utcnow(),
        estimated_cost=15.75,
        provider_job_url="https://mock.cloudtrain.ai/jobs/test-job-12345",
        metadata={
            "instance_type": "mock.cpu.small",
            "region": "mock-region-1",
            "test_mode": True,
        },
    )


@pytest.fixture
def sample_job_status_update():
    """Provide a sample job status update for testing."""
    return JobStatusUpdate(
        job_id="test-job-12345",
        status=JobStatus.RUNNING,
        progress_percentage=65.5,
        current_epoch=7,
        total_epochs=10,
        metrics={"loss": 0.25, "accuracy": 0.87, "learning_rate": 0.001},
        logs=[
            "[2025-01-01 12:00:00] Training started",
            "[2025-01-01 12:05:00] Epoch 1 completed",
            "[2025-01-01 12:10:00] Epoch 2 completed",
            "[2025-01-01 12:15:00] Current epoch: 7",
        ],
        error_message=None,
        updated_time=datetime.utcnow(),
    )


@pytest.fixture
def gpu_job_spec(sample_data_configuration, sample_environment_configuration):
    """Provide a GPU-based job specification for testing."""
    return TrainingJobSpec(
        job_name="gpu-training-job",
        description="GPU training job for testing",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL,
            instance_count=1,
            volume_size_gb=100,
            gpu_count=1,
            max_runtime_seconds=7200,
        ),
        data_configuration=sample_data_configuration,
        environment_configuration=sample_environment_configuration,
        tags={"gpu": "true", "test": "true"},
    )


@pytest.fixture
def custom_instance_job_spec(
    sample_data_configuration, sample_environment_configuration
):
    """Provide a custom instance type job specification for testing."""
    return TrainingJobSpec(
        job_name="custom-instance-job",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.CUSTOM,
            custom_instance_type="ml.p4d.24xlarge",
            instance_count=2,
            volume_size_gb=500,
        ),
        data_configuration=sample_data_configuration,
        environment_configuration=sample_environment_configuration,
    )


@pytest.fixture
def failed_job_status_update():
    """Provide a failed job status update for testing."""
    return JobStatusUpdate(
        job_id="failed-job-12345",
        status=JobStatus.FAILED,
        progress_percentage=45.0,
        current_epoch=4,
        total_epochs=10,
        metrics={"loss": 1.25, "accuracy": 0.45},
        logs=[
            "[2025-01-01 12:00:00] Training started",
            "[2025-01-01 12:05:00] Epoch 1 completed",
            "[2025-01-01 12:10:00] Epoch 2 completed",
            "[2025-01-01 12:15:00] Error occurred during epoch 4",
        ],
        error_message="CUDA out of memory error",
        updated_time=datetime.utcnow(),
    )


@pytest.fixture
def completed_job_status_update():
    """Provide a completed job status update for testing."""
    return JobStatusUpdate(
        job_id="completed-job-12345",
        status=JobStatus.COMPLETED,
        progress_percentage=100.0,
        current_epoch=10,
        total_epochs=10,
        metrics={"final_loss": 0.15, "final_accuracy": 0.94, "training_time": 3600.5},
        logs=[
            "[2025-01-01 12:00:00] Training started",
            "[2025-01-01 13:00:00] Training completed successfully",
            "[2025-01-01 13:00:01] Final metrics saved",
        ],
        error_message=None,
        updated_time=datetime.utcnow(),
    )


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")
    config.addinivalue_line("markers", "aws: marks tests that require AWS credentials")
    config.addinivalue_line(
        "markers", "azure: marks tests that require Azure credentials"
    )
    config.addinivalue_line("markers", "gcp: marks tests that require GCP credentials")


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(
            marker.name in ["integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to integration tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Add provider-specific markers based on test path
        test_path = str(item.fspath)
        if "aws" in test_path:
            item.add_marker(pytest.mark.aws)
        elif "azure" in test_path:
            item.add_marker(pytest.mark.azure)
        elif "gcp" in test_path:
            item.add_marker(pytest.mark.gcp)
