"""Mock cloud provider implementation for testing.

This module provides a mock implementation of the BaseCloudProvider
that simulates cloud training job operations without requiring actual
cloud provider credentials or resources.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.providers.base import BaseCloudProvider
from cloudtrain.schemas import JobStatusUpdate, TrainingJobResult, TrainingJobSpec

logger: logging.Logger = logging.getLogger(__name__)


class MockProvider(BaseCloudProvider):
    """Mock cloud provider for testing and development.

    This provider simulates the behavior of a real cloud provider
    without making actual API calls or requiring credentials.
    It's useful for testing, development, and demonstrations.

    Attributes:
        jobs: In-memory storage of submitted jobs
        job_progression: Simulated job status progression
    """

    def __init__(self, config_manager: Any) -> None:
        """Initialize the mock provider.

        Args:
            config_manager: Configuration manager instance
        """
        # In-memory job storage
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_progression: Dict[str, List[JobStatus]] = {}

        super().__init__(config_manager)

    def _get_provider_type(self) -> CloudProvider:
        """Get the provider type for this implementation.

        Returns:
            CloudProvider.MOCK
        """
        return CloudProvider.MOCK

    def _load_configuration(self) -> Dict[str, Any]:
        """Load mock provider configuration.

        Returns:
            Mock configuration dictionary
        """
        # Get mock provider config from config manager
        mock_config: Any = self.config_manager.get_provider_config(CloudProvider.MOCK)

        config: Dict[str, Any] = {
            "region": "mock-region-1",
            "endpoint": "https://mock.cloudtrain.ai",
            "api_version": "v1",
            "timeout": 30,
            "max_retries": 3,
        }

        # Add failure simulation settings if available
        if mock_config:
            config.update(
                {
                    "simulate_failures": getattr(
                        mock_config, "simulate_failures", False
                    ),
                    "failure_rate": getattr(mock_config, "failure_rate", 0.1),
                    "response_delay": getattr(mock_config, "response_delay", 0.1),
                }
            )

        return config

    async def _authenticate(self) -> None:
        """Simulate authentication with the mock provider.

        This method simulates the authentication process without
        making actual network calls.
        """
        # Simulate authentication delay
        await asyncio.sleep(0.1)

        logger.debug("Mock provider authentication successful")

    def _map_instance_type(
        self, instance_type: InstanceType, custom_type: Optional[str] = None
    ) -> str:
        """Map standardized instance type to mock provider type.

        Args:
            instance_type: Standardized instance type
            custom_type: Custom instance type string (if instance_type is CUSTOM)

        Returns:
            Mock provider-specific instance type string
        """
        if instance_type == InstanceType.CUSTOM:
            if not custom_type:
                raise ValueError("Custom instance type must be specified")
            return custom_type

        # Mock instance type mapping
        mapping: Dict[InstanceType, str] = {
            InstanceType.CPU_SMALL: "mock.cpu.small",
            InstanceType.CPU_MEDIUM: "mock.cpu.medium",
            InstanceType.CPU_LARGE: "mock.cpu.large",
            InstanceType.GPU_SMALL: "mock.gpu.small",
            InstanceType.GPU_MEDIUM: "mock.gpu.medium",
            InstanceType.GPU_LARGE: "mock.gpu.large",
        }

        if instance_type not in mapping:
            raise ValueError(f"Unsupported instance type: {instance_type}")

        return mapping[instance_type]

    def _map_job_status(self, provider_status: str) -> JobStatus:
        """Map mock provider job status to standardized status.

        Args:
            provider_status: Mock provider status string

        Returns:
            Standardized JobStatus enum value
        """
        # Mock status mapping
        mapping: Dict[str, JobStatus] = {
            "QUEUED": JobStatus.PENDING,
            "INITIALIZING": JobStatus.STARTING,
            "TRAINING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.COMPLETED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.STOPPED,
            "STOPPING": JobStatus.STOPPING,
        }

        return mapping.get(provider_status, JobStatus.UNKNOWN)

    def _simulate_job_progression(self, job_id: str) -> None:
        """Set up simulated job status progression.

        Args:
            job_id: Job identifier to set up progression for
        """
        # Define realistic job progression
        progression: List[JobStatus] = [
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,  # Could also be FAILED based on simulation
        ]

        self.job_progression[job_id] = progression

    def _get_current_job_status(self, job_id: str) -> JobStatus:
        """Get the current simulated status for a job.

        Args:
            job_id: Job identifier

        Returns:
            Current simulated job status
        """
        if job_id not in self.jobs:
            return JobStatus.UNKNOWN

        job_data: Dict[str, Any] = self.jobs[job_id]

        # Check if job has been manually set to a terminal status
        if "status" in job_data and job_data["status"].is_terminal():
            return job_data["status"]

        submission_time: datetime = job_data["submission_time"]
        elapsed: timedelta = datetime.now(UTC) - submission_time

        # Simulate job progression based on elapsed time
        if elapsed < timedelta(seconds=1):
            return JobStatus.PENDING
        elif elapsed < timedelta(seconds=2):
            return JobStatus.STARTING
        elif elapsed < timedelta(seconds=3):
            return JobStatus.RUNNING
        else:
            # Use configured failure simulation or deterministic success
            if self._config.get("simulate_failures", False):
                import random

                failure_rate: float = self._config.get("failure_rate", 0.1)
                if random.random() < failure_rate:
                    return JobStatus.FAILED
                else:
                    return JobStatus.COMPLETED
            else:
                # When failure simulation is disabled, always succeed for deterministic behavior
                return JobStatus.COMPLETED

    async def _submit_job_impl(self, job_spec: TrainingJobSpec) -> TrainingJobResult:
        """Mock job submission implementation.

        Args:
            job_spec: Complete training job specification

        Returns:
            Mock job submission result
        """
        # Simulate API call delay
        await asyncio.sleep(0.2)

        # Generate mock job ID
        job_id = f"mock-job-{uuid.uuid4().hex[:8]}"

        # Store job data
        submission_time = datetime.now(UTC)
        self.jobs[job_id] = {
            "job_spec": job_spec,
            "submission_time": submission_time,
            "status": JobStatus.PENDING,
        }

        # Set up job progression simulation
        self._simulate_job_progression(job_id)

        # Create result
        result = TrainingJobResult(
            job_id=job_id,
            job_name=job_spec.job_name,
            provider=CloudProvider.MOCK,
            status=JobStatus.PENDING,
            submission_time=submission_time,
            estimated_start_time=submission_time + timedelta(seconds=10),
            estimated_cost=12.50,  # Mock cost
            provider_job_url=f"https://mock.cloudtrain.ai/jobs/{job_id}",
            metadata={
                "mock_provider": True,
                "instance_type": self._map_instance_type(
                    job_spec.resource_requirements.instance_type,
                    job_spec.resource_requirements.custom_instance_type,
                ),
                "region": self._config["region"],
            },
        )

        logger.info(f"Mock job submitted: {job_id}")
        return result

    async def _get_job_status_impl(self, job_id: str) -> JobStatusUpdate:
        """Mock job status retrieval implementation.

        Args:
            job_id: Unique job identifier

        Returns:
            Mock job status update
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        current_status = self._get_current_job_status(job_id)
        job_data = self.jobs[job_id]

        # Update stored status
        self.jobs[job_id]["status"] = current_status

        # Simulate progress and metrics
        progress = None
        metrics = {}

        if current_status == JobStatus.RUNNING:
            elapsed = datetime.now(UTC) - job_data["submission_time"]
            progress = min(85.0, (elapsed.total_seconds() / 120.0) * 100)
            metrics = {
                "loss": 0.5 - (progress / 100) * 0.3,
                "accuracy": 0.6 + (progress / 100) * 0.35,
                "learning_rate": 0.001,
            }
        elif current_status == JobStatus.COMPLETED:
            progress = 100.0
            metrics = {
                "final_loss": 0.15,
                "final_accuracy": 0.92,
                "training_time": 118.5,
            }

        return JobStatusUpdate(
            job_id=job_id,
            status=current_status,
            progress_percentage=progress,
            current_epoch=int(progress / 10) if progress else None,
            total_epochs=10 if progress else None,
            metrics=metrics,
            logs=[
                f"[{datetime.now(UTC).isoformat()}] Job {job_id} status: {current_status.value}",
                (
                    f"[{datetime.now(UTC).isoformat()}] Progress: {progress}%"
                    if progress
                    else ""
                ),
            ],
            error_message=(
                "Simulated training error"
                if current_status == JobStatus.FAILED
                else None
            ),
            updated_time=datetime.now(UTC),
        )

    async def _cancel_job_impl(self, job_id: str) -> bool:
        """Mock job cancellation implementation.

        Args:
            job_id: Unique job identifier

        Returns:
            True if job was successfully cancelled
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        if job_id not in self.jobs:
            return False

        current_status = self._get_current_job_status(job_id)

        # Can only cancel active jobs
        if current_status.is_active():
            self.jobs[job_id]["status"] = JobStatus.STOPPED
            logger.info(f"Mock job cancelled: {job_id}")
            return True

        return False

    async def _list_jobs_impl(
        self, status_filter: Optional[JobStatus] = None, limit: int = 100
    ) -> List[JobStatusUpdate]:
        """Mock job listing implementation.

        Args:
            status_filter: Optional filter by job status
            limit: Maximum number of jobs to return

        Returns:
            List of mock job status updates
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        jobs = []
        count = 0

        for job_id in sorted(self.jobs.keys(), reverse=True):  # Most recent first
            if count >= limit:
                break

            current_status = self._get_current_job_status(job_id)

            if status_filter is None or current_status == status_filter:
                status_update = await self._get_job_status_impl(job_id)
                jobs.append(status_update)
                count += 1

        logger.debug(f"Mock provider returned {len(jobs)} jobs")
        return jobs
