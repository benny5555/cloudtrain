"""Base classes for cloud provider implementations.

This module defines the abstract base classes and interfaces that all
cloud provider implementations must follow in the CloudTrain system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.schemas import (
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)

logger: logging.Logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-specific errors."""

    def __init__(
        self, message: str, provider: CloudProvider, error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class AuthenticationError(ProviderError):
    """Exception raised when provider authentication fails."""

    pass


class ResourceNotAvailableError(ProviderError):
    """Exception raised when requested resources are not available."""

    pass


class JobSubmissionError(ProviderError):
    """Exception raised when job submission fails."""

    pass


class BaseCloudProvider(ABC):
    """Abstract base class for all cloud provider implementations.

    This class defines the interface that all cloud provider implementations
    must follow. It provides common functionality and enforces the contract
    for provider-specific operations.

    Attributes:
        provider_type: The cloud provider type this implementation supports
        config_manager: Configuration manager for credentials and settings
        is_authenticated: Whether the provider is properly authenticated
    """

    def __init__(self, config_manager: Any) -> None:
        """Initialize the base cloud provider.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager: Any = config_manager
        self.provider_type: CloudProvider = self._get_provider_type()
        self.is_authenticated: bool = False
        self._client: Optional[Any] = None

        # Initialize provider-specific configuration
        self._config: Dict[str, Any] = self._load_configuration()

        logger.debug(f"Initialized {self.provider_type.value} provider")

    @abstractmethod
    def _get_provider_type(self) -> CloudProvider:
        """Get the provider type for this implementation.

        Returns:
            The CloudProvider enum value for this implementation
        """
        pass

    @abstractmethod
    def _load_configuration(self) -> Dict[str, Any]:
        """Load provider-specific configuration.

        Returns:
            Dictionary containing provider configuration

        Raises:
            AuthenticationError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def _authenticate(self) -> None:
        """Authenticate with the cloud provider.

        This method should establish authentication with the provider
        and set up any necessary clients or sessions.

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    def _map_instance_type(
        self, instance_type: InstanceType, custom_type: Optional[str] = None
    ) -> str:
        """Map standardized instance type to provider-specific type.

        Args:
            instance_type: Standardized instance type
            custom_type: Custom instance type string (if instance_type is CUSTOM)

        Returns:
            Provider-specific instance type string

        Raises:
            ValueError: If instance type is not supported
        """
        pass

    @abstractmethod
    def _map_job_status(self, provider_status: str) -> JobStatus:
        """Map provider-specific job status to standardized status.

        Args:
            provider_status: Provider-specific status string

        Returns:
            Standardized JobStatus enum value
        """
        pass

    @abstractmethod
    async def _submit_job_impl(self, job_spec: TrainingJobSpec) -> TrainingJobResult:
        """Provider-specific job submission implementation.

        Args:
            job_spec: Complete training job specification

        Returns:
            Result of job submission

        Raises:
            JobSubmissionError: If job submission fails
        """
        pass

    @abstractmethod
    async def _get_job_status_impl(self, job_id: str) -> JobStatusUpdate:
        """Provider-specific job status retrieval implementation.

        Args:
            job_id: Unique job identifier

        Returns:
            Current job status and progress information

        Raises:
            ProviderError: If status retrieval fails
        """
        pass

    @abstractmethod
    async def _cancel_job_impl(self, job_id: str) -> bool:
        """Provider-specific job cancellation implementation.

        Args:
            job_id: Unique job identifier

        Returns:
            True if job was successfully cancelled

        Raises:
            ProviderError: If job cancellation fails
        """
        pass

    @abstractmethod
    async def _list_jobs_impl(
        self, status_filter: Optional[JobStatus] = None, limit: int = 100
    ) -> List[JobStatusUpdate]:
        """Provider-specific job listing implementation.

        Args:
            status_filter: Optional filter by job status
            limit: Maximum number of jobs to return

        Returns:
            List of job status updates

        Raises:
            ProviderError: If job listing fails
        """
        pass

    async def ensure_authenticated(self) -> None:
        """Ensure the provider is authenticated.

        This method checks if authentication is valid and re-authenticates
        if necessary.

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.is_authenticated:
            await self._authenticate()
            self.is_authenticated = True
            logger.info(f"Successfully authenticated with {self.provider_type.value}")

    def validate_job_spec(self, job_spec: TrainingJobSpec) -> None:
        """Validate job specification for this provider.

        Args:
            job_spec: Training job specification to validate

        Raises:
            ValueError: If job specification is invalid for this provider
        """
        # Base validation - can be overridden by providers
        if not job_spec.job_name:
            raise ValueError("Job name is required")

        if not job_spec.environment_configuration.entry_point:
            raise ValueError("Entry point is required")

        # Validate instance type mapping
        try:
            self._map_instance_type(
                job_spec.resource_requirements.instance_type,
                job_spec.resource_requirements.custom_instance_type,
            )
        except ValueError as e:
            raise ValueError(f"Invalid instance type: {e}")

    async def submit_job(self, job_spec: TrainingJobSpec) -> TrainingJobResult:
        """Submit a training job to this provider.

        Args:
            job_spec: Complete training job specification

        Returns:
            Result of job submission

        Raises:
            AuthenticationError: If authentication fails
            JobSubmissionError: If job submission fails
        """
        await self.ensure_authenticated()
        self.validate_job_spec(job_spec)

        try:
            result: TrainingJobResult = await self._submit_job_impl(job_spec)
            logger.info(
                f"Successfully submitted job {job_spec.job_name} "
                f"to {self.provider_type.value} with ID {result.job_id}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Failed to submit job {job_spec.job_name} "
                f"to {self.provider_type.value}: {e}"
            )
            raise JobSubmissionError(
                f"Job submission failed: {e}", self.provider_type
            ) from e

    async def get_job_status(self, job_id: str) -> JobStatusUpdate:
        """Get the status of a training job.

        Args:
            job_id: Unique job identifier

        Returns:
            Current job status and progress information

        Raises:
            AuthenticationError: If authentication fails
            ProviderError: If status retrieval fails
        """
        await self.ensure_authenticated()

        try:
            status: JobStatusUpdate = await self._get_job_status_impl(job_id)
            logger.debug(f"Retrieved status for job {job_id}: {status.status.value}")
            return status

        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {e}")
            raise ProviderError(
                f"Status retrieval failed: {e}", self.provider_type
            ) from e

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job.

        Args:
            job_id: Unique job identifier

        Returns:
            True if job was successfully cancelled

        Raises:
            AuthenticationError: If authentication fails
            ProviderError: If job cancellation fails
        """
        await self.ensure_authenticated()

        try:
            success: bool = await self._cancel_job_impl(job_id)
            if success:
                logger.info(f"Successfully cancelled job {job_id}")
            else:
                logger.warning(f"Job {job_id} could not be cancelled")
            return success

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise ProviderError(
                f"Job cancellation failed: {e}", self.provider_type
            ) from e

    async def list_jobs(
        self, status_filter: Optional[JobStatus] = None, limit: int = 100
    ) -> List[JobStatusUpdate]:
        """List training jobs for this provider.

        Args:
            status_filter: Optional filter by job status
            limit: Maximum number of jobs to return

        Returns:
            List of job status updates

        Raises:
            AuthenticationError: If authentication fails
            ProviderError: If job listing fails
        """
        await self.ensure_authenticated()

        try:
            jobs: List[JobStatusUpdate] = await self._list_jobs_impl(
                status_filter, limit
            )
            logger.debug(f"Retrieved {len(jobs)} jobs from {self.provider_type.value}")
            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs from {self.provider_type.value}: {e}")
            raise ProviderError(f"Job listing failed: {e}", self.provider_type) from e

    async def close(self) -> None:
        """Close provider connections and clean up resources.

        This method should be called when the provider is no longer needed
        to properly clean up any open connections or resources.
        """
        if self._client:
            try:
                # Close client connections if applicable
                if hasattr(self._client, "close"):
                    await self._client.close()
                elif hasattr(self._client, "__aexit__"):
                    await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    f"Error closing client for {self.provider_type.value}: {e}"
                )

        self.is_authenticated = False
        self._client = None
        logger.debug(f"Closed {self.provider_type.value} provider")

    def __str__(self) -> str:
        """Return string representation of the provider."""
        return f"{self.__class__.__name__}({self.provider_type.value})"

    def __repr__(self) -> str:
        """Return detailed string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider_type={self.provider_type.value}, "
            f"is_authenticated={self.is_authenticated})"
        )
