"""Main API class for CloudTrain universal cloud training.

This module provides the primary interface for submitting and managing
machine learning training jobs across multiple cloud providers.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Type, Union
import importlib

from cloudtrain.config import ConfigManager
from cloudtrain.enums import CloudProvider, JobStatus
from cloudtrain.providers.base import BaseCloudProvider
from cloudtrain.schemas import JobStatusUpdate, TrainingJobResult, TrainingJobSpec
from cloudtrain.utils.retry import retry_with_backoff
from cloudtrain.utils.validation import validate_job_spec

logger: logging.Logger = logging.getLogger(__name__)


class CloudTrainingAPI:
    """Main API class for universal cloud training.

    This class provides a unified interface for submitting machine learning
    training jobs to multiple cloud providers. It handles provider selection,
    credential management, job submission, and status monitoring.

    Example:
        Basic usage:

        >>> api = CloudTrainingAPI()
        >>> job_spec = TrainingJobSpec(
        ...     job_name="my-training-job",
        ...     resource_requirements=ResourceRequirements(
        ...         instance_type=InstanceType.GPU_SMALL
        ...     ),
        ...     data_configuration=DataConfiguration(
        ...         input_data_paths=["s3://bucket/data/"],
        ...         output_path="s3://bucket/output/"
        ...     ),
        ...     environment_configuration=EnvironmentConfiguration(
        ...         entry_point="train.py"
        ...     )
        ... )
        >>> result = await api.submit_job(CloudProvider.AWS, job_spec)
        >>> print(f"Job submitted: {result.job_id}")

    Attributes:
        config_manager: Configuration and credential manager
        providers: Registry of available cloud providers
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        auto_discover_providers: bool = True,
    ) -> None:
        """Initialize the CloudTrain API.

        Args:
            config_manager: Optional configuration manager instance
            auto_discover_providers: Whether to automatically discover available providers
        """
        self.config_manager = config_manager or ConfigManager()
        self.providers: Dict[CloudProvider, BaseCloudProvider] = {}

        if auto_discover_providers:
            self._discover_providers()

    def _discover_providers(self) -> None:
        """Discover and register available cloud providers.

        This method dynamically imports and registers provider implementations
        based on available dependencies and configuration.
        """
        # Import providers dynamically to avoid hard dependencies
        provider_modules: Dict[CloudProvider, str] = {
            CloudProvider.AWS: "cloudtrain.providers.aws.sagemaker",
            CloudProvider.AZURE: "cloudtrain.providers.azure.ml",
            CloudProvider.GCP: "cloudtrain.providers.gcp.aiplatform",
            CloudProvider.ALIBABA: "cloudtrain.providers.alibaba.pai",
            CloudProvider.TENCENT: "cloudtrain.providers.tencent.ti",
            CloudProvider.MOCK: "cloudtrain.providers.mock.provider",
        }

        for provider, module_name in provider_modules.items():
            try:
                # Dynamic import to handle optional dependencies
                module: Any = importlib.import_module(module_name)
                provider_class: Type[BaseCloudProvider] = getattr(
                    module, f"{provider.value.title()}Provider"
                )

                # Check if provider is properly configured
                if self._is_provider_configured(provider):
                    self.providers[provider] = provider_class(self.config_manager)
                    logger.info(f"Registered provider: {provider.value}")
                else:
                    logger.debug(f"Provider {provider.value} not configured, skipping")

            except ImportError as e:
                logger.debug(f"Provider {provider.value} not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to register provider {provider.value}: {e}")

    def _is_provider_configured(self, provider: CloudProvider) -> bool:
        """Check if a provider is properly configured.

        Args:
            provider: The cloud provider to check

        Returns:
            True if the provider has valid configuration
        """
        try:
            config: Optional[Any] = self.config_manager.get_provider_config(provider)
            return config is not None and config.is_valid()
        except Exception:
            return False

    def register_provider(
        self, provider: CloudProvider, provider_instance: BaseCloudProvider
    ) -> None:
        """Manually register a cloud provider.

        Args:
            provider: The cloud provider enum
            provider_instance: Instance of the provider implementation
        """
        self.providers[provider] = provider_instance
        logger.info(f"Manually registered provider: {provider.value}")

    def get_available_providers(self) -> List[CloudProvider]:
        """Get list of available and configured providers.

        Returns:
            List of cloud providers that are available for use
        """
        return list(self.providers.keys())

    async def submit_job(
        self, provider: CloudProvider, job_spec: TrainingJobSpec, dry_run: bool = False
    ) -> TrainingJobResult:
        """Submit a training job to the specified cloud provider.

        Args:
            provider: Cloud provider to submit the job to
            job_spec: Complete specification for the training job
            dry_run: If True, validate the job but don't actually submit

        Returns:
            Result of the job submission including job ID and status

        Raises:
            ValueError: If the provider is not available or job spec is invalid
            RuntimeError: If job submission fails
        """
        # Validate inputs
        if provider not in self.providers:
            available: List[str] = [p.value for p in self.get_available_providers()]
            raise ValueError(
                f"Provider {provider.value} is not available. "
                f"Available providers: {available}"
            )

        # Validate job specification
        validate_job_spec(job_spec)

        if dry_run:
            logger.info(f"Dry run: Job {job_spec.job_name} validated successfully")
            return TrainingJobResult(
                job_id="dry-run-job-id",
                job_name=job_spec.job_name,
                provider=provider,
                status=JobStatus.PENDING,
                submission_time=datetime.now(UTC),
            )

        # Get provider instance
        provider_instance: BaseCloudProvider = self.providers[provider]

        # Submit job with retry logic
        try:
            result: TrainingJobResult = await retry_with_backoff(
                provider_instance.submit_job, job_spec, max_retries=3, base_delay=1.0
            )

            logger.info(
                f"Successfully submitted job {job_spec.job_name} "
                f"to {provider.value} with ID {result.job_id}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Failed to submit job {job_spec.job_name} " f"to {provider.value}: {e}"
            )
            raise RuntimeError(f"Job submission failed: {e}") from e

    async def get_job_status(
        self, provider: CloudProvider, job_id: str
    ) -> JobStatusUpdate:
        """Get the current status of a training job.

        Args:
            provider: Cloud provider where the job is running
            job_id: Unique identifier of the job

        Returns:
            Current status and progress information for the job

        Raises:
            ValueError: If the provider is not available
            RuntimeError: If status retrieval fails
        """
        if provider not in self.providers:
            available: List[str] = [p.value for p in self.get_available_providers()]
            raise ValueError(
                f"Provider {provider.value} is not available. "
                f"Available providers: {available}"
            )

        provider_instance: BaseCloudProvider = self.providers[provider]

        try:
            status: JobStatusUpdate = await provider_instance.get_job_status(job_id)
            logger.debug(f"Retrieved status for job {job_id}: {status.status.value}")
            return status

        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {e}")
            raise RuntimeError(f"Status retrieval failed: {e}") from e

    async def cancel_job(self, provider: CloudProvider, job_id: str) -> bool:
        """Cancel a running training job.

        Args:
            provider: Cloud provider where the job is running
            job_id: Unique identifier of the job

        Returns:
            True if the job was successfully cancelled

        Raises:
            ValueError: If the provider is not available
            RuntimeError: If job cancellation fails
        """
        if provider not in self.providers:
            available: List[str] = [p.value for p in self.get_available_providers()]
            raise ValueError(
                f"Provider {provider.value} is not available. "
                f"Available providers: {available}"
            )

        provider_instance: BaseCloudProvider = self.providers[provider]

        try:
            success: bool = await provider_instance.cancel_job(job_id)
            if success:
                logger.info(f"Successfully cancelled job {job_id}")
            else:
                logger.warning(f"Job {job_id} could not be cancelled")
            return success

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise RuntimeError(f"Job cancellation failed: {e}") from e

    async def list_jobs(
        self,
        provider: CloudProvider,
        status_filter: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[JobStatusUpdate]:
        """List training jobs for a provider.

        Args:
            provider: Cloud provider to query
            status_filter: Optional filter by job status
            limit: Maximum number of jobs to return

        Returns:
            List of job status updates

        Raises:
            ValueError: If the provider is not available
        """
        if provider not in self.providers:
            available: List[str] = [p.value for p in self.get_available_providers()]
            raise ValueError(
                f"Provider {provider.value} is not available. "
                f"Available providers: {available}"
            )

        provider_instance: BaseCloudProvider = self.providers[provider]

        try:
            jobs: List[JobStatusUpdate] = await provider_instance.list_jobs(
                status_filter, limit
            )
            logger.debug(f"Retrieved {len(jobs)} jobs from {provider.value}")
            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs from {provider.value}: {e}")
            raise RuntimeError(f"Job listing failed: {e}") from e

    async def close(self) -> None:
        """Close all provider connections and clean up resources."""
        for provider_name, provider_instance in self.providers.items():
            try:
                await provider_instance.close()
                logger.debug(f"Closed provider: {provider_name.value}")
            except Exception as e:
                logger.warning(f"Error closing provider {provider_name.value}: {e}")

        self.providers.clear()
        logger.info("CloudTrain API closed successfully")

    async def __aenter__(self) -> "CloudTrainingAPI":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit."""
        await self.close()
