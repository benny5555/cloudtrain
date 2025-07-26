"""Unit tests for CloudTrain enums."""

import pytest

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus, LogLevel


class TestCloudProvider:
    """Test CloudProvider enum."""

    def test_provider_values(self):
        """Test that provider values are correct."""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.ALIBABA.value == "alibaba"
        assert CloudProvider.TENCENT.value == "tencent"
        assert CloudProvider.MOCK.value == "mock"

    def test_provider_string_representation(self):
        """Test string representation of providers."""
        assert str(CloudProvider.AWS) == "aws"
        assert str(CloudProvider.AZURE) == "azure"
        assert str(CloudProvider.MOCK) == "mock"

    def test_get_native_api_providers(self):
        """Test getting native API providers."""
        native_providers = CloudProvider.get_native_api_providers()

        expected = {
            CloudProvider.AWS,
            CloudProvider.AZURE,
            CloudProvider.GCP,
            CloudProvider.ALIBABA,
            CloudProvider.TENCENT,
        }

        assert native_providers == expected
        assert CloudProvider.MOCK not in native_providers

    def test_get_wrapper_providers(self):
        """Test getting wrapper providers."""
        wrapper_providers = CloudProvider.get_wrapper_providers()

        assert CloudProvider.MOCK in wrapper_providers
        assert CloudProvider.AWS not in wrapper_providers


class TestJobStatus:
    """Test JobStatus enum."""

    def test_status_values(self):
        """Test that status values are correct."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.STARTING.value == "starting"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.STOPPED.value == "stopped"
        assert JobStatus.STOPPING.value == "stopping"
        assert JobStatus.UNKNOWN.value == "unknown"

    def test_get_terminal_statuses(self):
        """Test getting terminal statuses."""
        terminal_statuses = JobStatus.get_terminal_statuses()

        expected = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}
        assert terminal_statuses == expected

    def test_get_active_statuses(self):
        """Test getting active statuses."""
        active_statuses = JobStatus.get_active_statuses()

        expected = {
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.STOPPING,
        }
        assert active_statuses == expected

    def test_is_terminal(self):
        """Test terminal status identification."""
        # Terminal statuses
        assert JobStatus.COMPLETED.is_terminal()
        assert JobStatus.FAILED.is_terminal()
        assert JobStatus.STOPPED.is_terminal()

        # Non-terminal statuses
        assert not JobStatus.PENDING.is_terminal()
        assert not JobStatus.RUNNING.is_terminal()
        assert not JobStatus.STARTING.is_terminal()
        assert not JobStatus.STOPPING.is_terminal()

    def test_is_active(self):
        """Test active status identification."""
        # Active statuses
        assert JobStatus.PENDING.is_active()
        assert JobStatus.STARTING.is_active()
        assert JobStatus.RUNNING.is_active()
        assert JobStatus.STOPPING.is_active()

        # Inactive statuses
        assert not JobStatus.COMPLETED.is_active()
        assert not JobStatus.FAILED.is_active()
        assert not JobStatus.STOPPED.is_active()

    def test_status_string_representation(self):
        """Test string representation of statuses."""
        assert str(JobStatus.RUNNING) == "running"
        assert str(JobStatus.COMPLETED) == "completed"
        assert str(JobStatus.FAILED) == "failed"


class TestInstanceType:
    """Test InstanceType enum."""

    def test_instance_type_values(self):
        """Test that instance type values are correct."""
        assert InstanceType.CPU_SMALL.value == "cpu_small"
        assert InstanceType.CPU_MEDIUM.value == "cpu_medium"
        assert InstanceType.CPU_LARGE.value == "cpu_large"
        assert InstanceType.GPU_SMALL.value == "gpu_small"
        assert InstanceType.GPU_MEDIUM.value == "gpu_medium"
        assert InstanceType.GPU_LARGE.value == "gpu_large"
        assert InstanceType.CUSTOM.value == "custom"

    def test_has_gpu(self):
        """Test GPU detection for instance types."""
        # GPU instance types
        assert InstanceType.GPU_SMALL.has_gpu()
        assert InstanceType.GPU_MEDIUM.has_gpu()
        assert InstanceType.GPU_LARGE.has_gpu()

        # CPU instance types
        assert not InstanceType.CPU_SMALL.has_gpu()
        assert not InstanceType.CPU_MEDIUM.has_gpu()
        assert not InstanceType.CPU_LARGE.has_gpu()

        # Custom instance type
        assert not InstanceType.CUSTOM.has_gpu()

    def test_instance_type_string_representation(self):
        """Test string representation of instance types."""
        assert str(InstanceType.CPU_SMALL) == "cpu_small"
        assert str(InstanceType.GPU_LARGE) == "gpu_large"
        assert str(InstanceType.CUSTOM) == "custom"


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self):
        """Test that log level values are correct."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_log_level_string_representation(self):
        """Test string representation of log levels."""
        assert str(LogLevel.DEBUG) == "debug"
        assert str(LogLevel.INFO) == "info"
        assert str(LogLevel.ERROR) == "error"


@pytest.mark.unit
class TestEnumIntegration:
    """Test enum integration and edge cases."""

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert CloudProvider.AWS in CloudProvider
        assert JobStatus.RUNNING in JobStatus
        assert InstanceType.GPU_SMALL in InstanceType
        assert LogLevel.INFO in LogLevel

    def test_enum_iteration(self):
        """Test enum iteration."""
        cloud_providers = list(CloudProvider)
        assert len(cloud_providers) == 6  # AWS, AZURE, GCP, ALIBABA, TENCENT, MOCK

        job_statuses = list(JobStatus)
        assert len(job_statuses) == 8  # All defined statuses

        instance_types = list(InstanceType)
        assert len(instance_types) == 7  # All defined instance types

        log_levels = list(LogLevel)
        assert len(log_levels) == 5  # All defined log levels

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        # Enums should be comparable for equality
        assert CloudProvider.AWS == CloudProvider.AWS
        assert CloudProvider.AWS != CloudProvider.AZURE

        assert JobStatus.RUNNING == JobStatus.RUNNING
        assert JobStatus.RUNNING != JobStatus.COMPLETED

    def test_enum_hashing(self):
        """Test that enums are hashable and can be used in sets/dicts."""
        provider_set = {CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.AWS}
        assert len(provider_set) == 2  # Duplicates removed

        status_dict = {
            JobStatus.RUNNING: "active",
            JobStatus.COMPLETED: "finished",
            JobStatus.FAILED: "error",
        }
        assert len(status_dict) == 3
        assert status_dict[JobStatus.RUNNING] == "active"
