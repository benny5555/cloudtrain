#!/usr/bin/env python3
"""Multi-provider example for CloudTrain.

This example demonstrates how to use CloudTrain with multiple cloud providers,
including provider selection, failover strategies, and cost optimization.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Union, cast

from cloudtrain import (
    CloudProvider,
    CloudTrainingAPI,
    DataConfiguration,
    EnvironmentConfiguration,
    InstanceType,
    JobStatus,
    ResourceRequirements,
    TrainingJobSpec,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiProviderTrainingManager:
    """Manager for handling training jobs across multiple providers."""

    def __init__(self):
        self.api = None
        self.provider_preferences = [
            CloudProvider.AWS,
            CloudProvider.AZURE,
            CloudProvider.GCP,
            CloudProvider.MOCK,  # Fallback for testing
        ]

    async def __aenter__(self):
        self.api = CloudTrainingAPI()
        await self.api.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.api:
            await self.api.__aexit__(exc_type, exc_val, exc_tb)

    def get_available_providers(self) -> List[CloudProvider]:
        """Get list of available providers in preference order."""
        available = self.api.get_available_providers()

        # Return providers in preference order
        ordered_providers = []
        for provider in self.provider_preferences:
            if provider in available:
                ordered_providers.append(provider)

        # Add any remaining providers
        for provider in available:
            if provider not in ordered_providers:
                ordered_providers.append(provider)

        return ordered_providers

    async def submit_with_failover(
        self,
        job_spec: TrainingJobSpec,
        preferred_providers: Optional[List[CloudProvider]] = None,
    ) -> Tuple[CloudProvider, str]:
        """Submit job with automatic failover to alternative providers."""

        providers_to_try = preferred_providers or self.get_available_providers()

        last_error = None

        for provider in providers_to_try:
            try:
                logger.info(f"Attempting to submit job to {provider.value}...")

                result = await self.api.submit_job(provider, job_spec)

                logger.info(f"Successfully submitted job to {provider.value}")
                logger.info(f"Job ID: {result.job_id}")

                return provider, result.job_id

            except Exception as e:
                logger.warning(f"Failed to submit to {provider.value}: {e}")
                last_error = e
                continue

        # All providers failed
        raise RuntimeError(
            f"Failed to submit job to any provider. Last error: {last_error}"
        )

    async def submit_parallel_jobs(
        self, job_specs: List[Tuple[TrainingJobSpec, Optional[CloudProvider]]]
    ) -> List[Tuple[Optional[CloudProvider], str, bool]]:
        """Submit multiple jobs in parallel across different providers."""

        async def submit_single_job(job_spec, preferred_provider):
            try:
                if (
                    preferred_provider
                    and preferred_provider in self.get_available_providers()
                ):
                    result = await self.api.submit_job(preferred_provider, job_spec)
                    return preferred_provider, result.job_id, True
                else:
                    provider, job_id = await self.submit_with_failover(job_spec)
                    return provider, job_id, True
            except Exception as e:
                logger.error(f"Failed to submit job {job_spec.job_name}: {e}")
                return None, str(e), False

        # Submit all jobs concurrently
        tasks = [
            submit_single_job(job_spec, preferred_provider)
            for job_spec, preferred_provider in job_specs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results: List[Tuple[Optional[CloudProvider], str, bool]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                job_name = job_specs[i][0].job_name
                logger.error(f"Job {job_name} failed with exception: {result}")
                processed_results.append((None, str(result), False))
            else:
                # result is guaranteed to be the correct tuple type here
                processed_results.append(
                    cast(Tuple[Optional[CloudProvider], str, bool], result)
                )

        return processed_results

    async def monitor_multiple_jobs(
        self, jobs: List[Tuple[CloudProvider, str]]
    ) -> List[JobStatus]:
        """Monitor multiple jobs across different providers."""

        async def monitor_single_job(provider, job_id):
            try:
                status = await self.api.get_job_status(provider, job_id)
                return status.status
            except Exception as e:
                logger.error(
                    f"Failed to get status for job {job_id} on {provider.value}: {e}"
                )
                return JobStatus.UNKNOWN

        # Get status for all jobs concurrently
        tasks = [monitor_single_job(provider, job_id) for provider, job_id in jobs]
        statuses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to UNKNOWN status
        final_statuses: List[JobStatus] = []
        for status in statuses:
            if isinstance(status, Exception):
                final_statuses.append(JobStatus.UNKNOWN)
            else:
                # status is guaranteed to be JobStatus here
                final_statuses.append(cast(JobStatus, status))

        return final_statuses


async def demonstrate_provider_selection():
    """Demonstrate intelligent provider selection."""

    logger.info("Demonstrating provider selection...")

    async with MultiProviderTrainingManager() as manager:

        available_providers = manager.get_available_providers()
        logger.info(f"Available providers: {[p.value for p in available_providers]}")

        # Create a sample job
        job_spec = TrainingJobSpec(
            job_name="provider-selection-test",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_MEDIUM
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["s3://example-bucket/data/"],
                output_path="s3://example-bucket/output/",
            ),
            environment_configuration=EnvironmentConfiguration(
                entry_point="train.py",
                framework="tensorflow",
                framework_version="2.12.0",
            ),
        )

        # Submit with automatic provider selection
        try:
            provider, job_id = await manager.submit_with_failover(job_spec)
            logger.info(f"Job submitted to {provider.value} with ID: {job_id}")
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")


async def demonstrate_parallel_submission():
    """Demonstrate parallel job submission across providers."""

    logger.info("Demonstrating parallel job submission...")

    async with MultiProviderTrainingManager() as manager:

        # Create multiple job specifications
        job_specs = []

        for i in range(3):
            job_spec = TrainingJobSpec(
                job_name=f"parallel-job-{i+1}",
                resource_requirements=ResourceRequirements(
                    instance_type=InstanceType.CPU_SMALL
                ),
                data_configuration=DataConfiguration(
                    input_data_paths=[f"s3://example-bucket/data-{i+1}/"],
                    output_path=f"s3://example-bucket/output-{i+1}/",
                ),
                environment_configuration=EnvironmentConfiguration(
                    entry_point="train.py",
                    environment_variables={"JOB_INDEX": str(i + 1)},
                ),
            )

            # Specify preferred provider (None for automatic selection)
            preferred_provider = None
            if i == 0:
                preferred_provider = CloudProvider.MOCK  # Force first job to mock

            job_specs.append((job_spec, preferred_provider))

        # Submit all jobs in parallel
        results = await manager.submit_parallel_jobs(job_specs)

        # Display results
        successful_jobs = []
        for i, (provider, job_id, success) in enumerate(results):
            if success and provider is not None:
                logger.info(
                    f"Job {i+1}: Submitted to {provider.value} with ID {job_id}"
                )
                successful_jobs.append((provider, job_id))
            else:
                logger.error(f"Job {i+1}: Failed - {job_id}")

        return successful_jobs


async def demonstrate_multi_provider_monitoring():
    """Demonstrate monitoring jobs across multiple providers."""

    logger.info("Demonstrating multi-provider monitoring...")

    async with MultiProviderTrainingManager() as manager:

        # Submit a few test jobs
        jobs = await demonstrate_parallel_submission()

        if not jobs:
            logger.warning("No jobs to monitor")
            return

        logger.info(f"Monitoring {len(jobs)} jobs across providers...")

        # Monitor jobs until completion
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            statuses = await manager.monitor_multiple_jobs(jobs)

            logger.info(f"Monitoring iteration {iteration + 1}:")

            all_complete = True
            for i, ((provider, job_id), status) in enumerate(zip(jobs, statuses)):
                # provider is guaranteed to be non-None from successful_jobs
                assert provider is not None
                logger.info(f"  Job {i+1} ({provider.value}): {status.value}")

                if not status.is_terminal():
                    all_complete = False

            if all_complete:
                logger.info("All jobs completed!")
                break

            iteration += 1
            await asyncio.sleep(5)

        if iteration >= max_iterations:
            logger.warning("Monitoring timeout reached")


async def demonstrate_cost_optimization():
    """Demonstrate cost optimization across providers."""

    logger.info("Demonstrating cost optimization...")

    # This is a conceptual example - actual cost optimization would require
    # real provider pricing APIs and more sophisticated logic

    job_spec = TrainingJobSpec(
        job_name="cost-optimization-test",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL, max_runtime_seconds=1800  # 30 minutes
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["s3://example-bucket/data/"],
            output_path="s3://example-bucket/output/",
        ),
        environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
    )

    # Simulated cost estimates (in practice, these would come from provider APIs)
    cost_estimates = {
        CloudProvider.AWS: 2.50,
        CloudProvider.AZURE: 2.30,
        CloudProvider.GCP: 2.40,
        CloudProvider.MOCK: 0.00,  # Free for testing
    }

    async with MultiProviderTrainingManager() as manager:
        available_providers = manager.get_available_providers()

        # Filter available providers and sort by cost
        available_costs = {
            provider: cost_estimates.get(provider, float("inf"))
            for provider in available_providers
        }

        sorted_providers = sorted(available_costs.items(), key=lambda x: x[1])

        logger.info("Cost optimization analysis:")
        for provider, cost in sorted_providers:
            logger.info(f"  {provider.value}: ${cost:.2f}")

        # Select the cheapest available provider
        cheapest_provider = sorted_providers[0][0]
        logger.info(f"Selected cheapest provider: {cheapest_provider.value}")

        try:
            result = await manager.api.submit_job(cheapest_provider, job_spec)
            logger.info(f"Cost-optimized job submitted: {result.job_id}")
        except Exception as e:
            logger.error(f"Failed to submit to cheapest provider: {e}")
            logger.info("Falling back to failover strategy...")

            provider, job_id = await manager.submit_with_failover(
                job_spec,
                [p for p, _ in sorted_providers[1:]],  # Try remaining providers
            )
            logger.info(f"Fallback job submitted to {provider.value}: {job_id}")


async def main():
    """Run all multi-provider examples."""

    logger.info("=" * 60)
    logger.info("CloudTrain Multi-Provider Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Provider selection
        logger.info("\n1. Provider Selection")
        logger.info("-" * 30)
        await demonstrate_provider_selection()

        # Example 2: Parallel submission
        logger.info("\n2. Parallel Job Submission")
        logger.info("-" * 30)
        await demonstrate_parallel_submission()

        # Example 3: Multi-provider monitoring
        logger.info("\n3. Multi-Provider Monitoring")
        logger.info("-" * 30)
        await demonstrate_multi_provider_monitoring()

        # Example 4: Cost optimization
        logger.info("\n4. Cost Optimization")
        logger.info("-" * 30)
        await demonstrate_cost_optimization()

        logger.info("\n" + "=" * 60)
        logger.info("All multi-provider examples completed! 🎉")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Multi-provider example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
