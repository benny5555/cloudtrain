"""Unit tests for CloudTrain enums."""

from typing import Set

import pytest

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus, LogLevel


class TestCloudProvider:
    """Test CloudProvider enum."""

    @pytest.mark.parametrize(
        "provider,expected_value",
        [
            (CloudProvider.AWS, "aws"),
            (CloudProvider.AZURE, "azure"),
            (CloudProvider.GCP, "gcp"),
            (CloudProvider.ALIBABA, "alibaba"),
            (CloudProvider.TENCENT, "tencent"),
            (CloudProvider.MOCK, "mock"),
        ],
    )
    def test_provider_values_and_string_representation(
        self, provider: CloudProvider, expected_value: str
    ) -> None:
        """Test that provider values and string representations are correct."""
        assert provider.value == expected_value
        assert str(provider) == expected_value

    def test_get_native_api_providers(self) -> None:
        """Test getting native API providers."""
        native_providers: Set[CloudProvider] = CloudProvider.get_native_api_providers()

        expected: Set[CloudProvider] = {
            CloudProvider.AWS,
            CloudProvider.AZURE,
            CloudProvider.GCP,
            CloudProvider.ALIBABA,
            CloudProvider.TENCENT,
        }

        assert native_providers == expected
        assert CloudProvider.MOCK not in native_providers

    def test_get_wrapper_providers(self) -> None:
        """Test getting wrapper providers."""
        wrapper_providers: Set[CloudProvider] = CloudProvider.get_wrapper_providers()

        assert CloudProvider.MOCK in wrapper_providers
        assert CloudProvider.AWS not in wrapper_providers


class TestJobStatus:
    """Test JobStatus enum."""

    @pytest.mark.parametrize(
        "status,expected_value",
        [
            (JobStatus.PENDING, "pending"),
            (JobStatus.STARTING, "starting"),
            (JobStatus.RUNNING, "running"),
            (JobStatus.COMPLETED, "completed"),
            (JobStatus.FAILED, "failed"),
            (JobStatus.STOPPED, "stopped"),
            (JobStatus.STOPPING, "stopping"),
            (JobStatus.UNKNOWN, "unknown"),
        ],
    )
    def test_status_values(self, status, expected_value):
        """Test that status values are correct."""
        assert status.value == expected_value

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

    @pytest.mark.parametrize(
        "status,is_terminal",
        [
            (JobStatus.COMPLETED, True),
            (JobStatus.FAILED, True),
            (JobStatus.STOPPED, True),
            (JobStatus.PENDING, False),
            (JobStatus.RUNNING, False),
            (JobStatus.STARTING, False),
            (JobStatus.STOPPING, False),
            (JobStatus.UNKNOWN, False),
        ],
    )
    def test_is_terminal(self, status, is_terminal):
        """Test terminal status identification."""
        assert status.is_terminal() == is_terminal

    @pytest.mark.parametrize(
        "status,is_active",
        [
            (JobStatus.PENDING, True),
            (JobStatus.STARTING, True),
            (JobStatus.RUNNING, True),
            (JobStatus.STOPPING, True),
            (JobStatus.COMPLETED, False),
            (JobStatus.FAILED, False),
            (JobStatus.STOPPED, False),
            (JobStatus.UNKNOWN, False),
        ],
    )
    def test_is_active(self, status, is_active):
        """Test active status identification."""
        assert status.is_active() == is_active


class TestInstanceType:
    """Test InstanceType enum."""

    @pytest.mark.parametrize(
        "instance_type,expected_value,has_gpu",
        [
            (InstanceType.CPU_SMALL, "cpu_small", False),
            (InstanceType.CPU_MEDIUM, "cpu_medium", False),
            (InstanceType.CPU_LARGE, "cpu_large", False),
            (InstanceType.GPU_SMALL, "gpu_small", True),
            (InstanceType.GPU_MEDIUM, "gpu_medium", True),
            (InstanceType.GPU_LARGE, "gpu_large", True),
            (InstanceType.CUSTOM, "custom", False),
        ],
    )
    def test_instance_type_properties(self, instance_type, expected_value, has_gpu):
        """Test instance type values, string representation, and GPU detection."""
        assert instance_type.value == expected_value
        assert str(instance_type) == expected_value
        assert instance_type.has_gpu() == has_gpu


class TestLogLevel:
    """Test LogLevel enum."""

    @pytest.mark.parametrize(
        "log_level,expected_value",
        [
            (LogLevel.DEBUG, "debug"),
            (LogLevel.INFO, "info"),
            (LogLevel.WARNING, "warning"),
            (LogLevel.ERROR, "error"),
            (LogLevel.CRITICAL, "critical"),
        ],
    )
    def test_log_level_values_and_string_representation(
        self, log_level, expected_value
    ):
        """Test log level values and string representations are correct."""
        assert log_level.value == expected_value
        assert str(log_level) == expected_value


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
