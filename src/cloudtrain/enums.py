"""Enumerations for CloudTrain API.

This module defines the core enumerations used throughout the CloudTrain
universal cloud training API, including cloud providers, job statuses,
and other categorical values.
"""

from enum import Enum
from typing import Set


class CloudProvider(Enum):
    """Enumeration of supported cloud providers.

    This enum provides type-safe identification of cloud providers
    supported by CloudTrain. Each provider has a string value that
    can be used for configuration and logging.

    Attributes:
        AWS: Amazon Web Services (SageMaker)
        AZURE: Microsoft Azure (Azure Machine Learning)
        GCP: Google Cloud Platform (AI Platform)
        ALIBABA: Alibaba Cloud (PAI)
        TENCENT: Tencent Cloud (TI)
        MOCK: Mock provider for testing
    """

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    MOCK = "mock"

    @classmethod
    def get_native_api_providers(cls) -> Set["CloudProvider"]:
        """Get providers that have native APIs.

        Returns:
            Set of providers with native ML training APIs.
        """
        return {cls.AWS, cls.AZURE, cls.GCP, cls.ALIBABA, cls.TENCENT}

    @classmethod
    def get_wrapper_providers(cls) -> Set["CloudProvider"]:
        """Get providers that require wrapper implementations.

        Returns:
            Set of providers requiring custom wrapper solutions.
        """
        return {cls.MOCK}  # Will expand as we add more providers

    def __str__(self) -> str:
        """Return the string representation of the provider."""
        return self.value


class JobStatus(Enum):
    """Enumeration of training job statuses.

    This enum represents the possible states of a training job
    across all supported cloud providers. The statuses are
    normalized to provide a consistent interface.

    Attributes:
        PENDING: Job is queued but not yet started
        STARTING: Job is initializing resources
        RUNNING: Job is actively training
        COMPLETED: Job finished successfully
        FAILED: Job failed with an error
        STOPPED: Job was manually stopped
        STOPPING: Job is in the process of stopping
        UNKNOWN: Job status cannot be determined
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

    @classmethod
    def get_terminal_statuses(cls) -> Set["JobStatus"]:
        """Get statuses that indicate job completion.

        Returns:
            Set of statuses indicating the job has finished.
        """
        return {cls.COMPLETED, cls.FAILED, cls.STOPPED}

    @classmethod
    def get_active_statuses(cls) -> Set["JobStatus"]:
        """Get statuses that indicate job is active.

        Returns:
            Set of statuses indicating the job is still running.
        """
        return {cls.PENDING, cls.STARTING, cls.RUNNING, cls.STOPPING}

    def is_terminal(self) -> bool:
        """Check if this status indicates job completion.

        Returns:
            True if the job has finished (success or failure).
        """
        return self in self.get_terminal_statuses()

    def is_active(self) -> bool:
        """Check if this status indicates job is still active.

        Returns:
            True if the job is still running or transitioning.
        """
        return self in self.get_active_statuses()

    def __str__(self) -> str:
        """Return the string representation of the status."""
        return self.value


class InstanceType(Enum):
    """Enumeration of standardized instance types.

    This enum provides a normalized view of compute instance types
    across different cloud providers. Each provider maps these
    to their specific instance types.

    Attributes:
        CPU_SMALL: Small CPU-only instance
        CPU_MEDIUM: Medium CPU-only instance
        CPU_LARGE: Large CPU-only instance
        GPU_SMALL: Small GPU instance (1 GPU)
        GPU_MEDIUM: Medium GPU instance (2-4 GPUs)
        GPU_LARGE: Large GPU instance (8+ GPUs)
        CUSTOM: Custom instance type (provider-specific)
    """

    CPU_SMALL = "cpu_small"
    CPU_MEDIUM = "cpu_medium"
    CPU_LARGE = "cpu_large"
    GPU_SMALL = "gpu_small"
    GPU_MEDIUM = "gpu_medium"
    GPU_LARGE = "gpu_large"
    CUSTOM = "custom"

    def has_gpu(self) -> bool:
        """Check if this instance type includes GPU resources.

        Returns:
            True if the instance type includes GPU resources.
        """
        return self.value.startswith("gpu")

    def __str__(self) -> str:
        """Return the string representation of the instance type."""
        return self.value


class LogLevel(Enum):
    """Enumeration of logging levels.

    This enum defines the available logging levels for CloudTrain
    operations and job execution.

    Attributes:
        DEBUG: Detailed debugging information
        INFO: General information messages
        WARNING: Warning messages for potential issues
        ERROR: Error messages for failures
        CRITICAL: Critical error messages
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __str__(self) -> str:
        """Return the string representation of the log level."""
        return self.value
