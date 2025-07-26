#!/usr/bin/env python3
"""Basic usage example for CloudTrain.

This example demonstrates the fundamental usage of CloudTrain for submitting
and monitoring machine learning training jobs across different cloud providers.
"""

import asyncio
import logging
from pathlib import Path

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


async def basic_job_submission():
    """Demonstrate basic job submission to the mock provider."""

    logger.info("Starting basic CloudTrain example...")

    # Initialize the CloudTrain API
    async with CloudTrainingAPI() as api:

        # Check available providers
        providers = api.get_available_providers()
        logger.info(f"Available providers: {[p.value for p in providers]}")

        # Create a training job specification
        job_spec = TrainingJobSpec(
            job_name="basic-pytorch-training",
            description="Basic PyTorch training job example",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.GPU_SMALL,
                instance_count=1,
                volume_size_gb=50,
                max_runtime_seconds=3600,  # 1 hour
            ),
            data_configuration=DataConfiguration(
                input_data_paths=[
                    "s3://my-training-bucket/datasets/cifar10/",
                    "s3://my-training-bucket/datasets/labels/",
                ],
                output_path="s3://my-training-bucket/outputs/basic-training/",
                checkpoint_path="s3://my-training-bucket/checkpoints/basic-training/",
                data_format="pytorch",
            ),
            environment_configuration=EnvironmentConfiguration(
                entry_point="train.py",
                framework="pytorch",
                framework_version="2.0.0",
                python_version="3.9",
                environment_variables={
                    "EPOCHS": "10",
                    "BATCH_SIZE": "32",
                    "LEARNING_RATE": "0.001",
                },
                command_line_args=["--model", "resnet18", "--optimizer", "adam"],
            ),
            tags={
                "project": "image-classification",
                "team": "ml-research",
                "environment": "development",
            },
        )

        # Submit the job to the mock provider (always available)
        logger.info("Submitting job to mock provider...")
        result = await api.submit_job(CloudProvider.MOCK, job_spec)

        logger.info(f"Job submitted successfully!")
        logger.info(f"  Job ID: {result.job_id}")
        logger.info(f"  Job Name: {result.job_name}")
        logger.info(f"  Provider: {result.provider.value}")
        logger.info(f"  Status: {result.status.value}")
        logger.info(f"  Submission Time: {result.submission_time}")
        logger.info(f"  Estimated Cost: ${result.estimated_cost:.2f}")

        if result.provider_job_url:
            logger.info(f"  Job URL: {result.provider_job_url}")

        return result.job_id


async def monitor_job_progress(api: CloudTrainingAPI, job_id: str):
    """Monitor job progress until completion."""

    logger.info(f"Monitoring job {job_id}...")

    while True:
        # Get current job status
        status = await api.get_job_status(CloudProvider.MOCK, job_id)

        logger.info(f"Job Status: {status.status.value}")

        if status.progress_percentage is not None:
            logger.info(f"Progress: {status.progress_percentage:.1f}%")

        if status.current_epoch is not None and status.total_epochs is not None:
            logger.info(f"Epoch: {status.current_epoch}/{status.total_epochs}")

        if status.metrics:
            metrics_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in status.metrics.items()]
            )
            logger.info(f"Metrics: {metrics_str}")

        # Show recent logs
        if status.logs:
            logger.info("Recent logs:")
            for log_line in status.logs[-3:]:  # Show last 3 log lines
                if log_line.strip():
                    logger.info(f"  {log_line}")

        # Check if job is finished
        if status.status.is_terminal():
            logger.info(f"Job finished with status: {status.status.value}")

            if status.status == JobStatus.COMPLETED:
                logger.info("Job completed successfully! 🎉")
                if status.metrics:
                    final_metrics = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in status.metrics.items()]
                    )
                    logger.info(f"Final metrics: {final_metrics}")

            elif status.status == JobStatus.FAILED:
                logger.error("Job failed! ❌")
                if status.error_message:
                    logger.error(f"Error: {status.error_message}")

            elif status.status == JobStatus.STOPPED:
                logger.warning("Job was stopped/cancelled")

            break

        # Wait before next status check
        await asyncio.sleep(10)


async def demonstrate_job_listing():
    """Demonstrate listing jobs from a provider."""

    logger.info("Demonstrating job listing...")

    async with CloudTrainingAPI() as api:

        # List all jobs
        all_jobs = await api.list_jobs(CloudProvider.MOCK, limit=10)
        logger.info(f"Found {len(all_jobs)} total jobs")

        # List only running jobs
        running_jobs = await api.list_jobs(
            CloudProvider.MOCK, status_filter=JobStatus.RUNNING, limit=5
        )
        logger.info(f"Found {len(running_jobs)} running jobs")

        # Display job information
        for job in all_jobs[:3]:  # Show first 3 jobs
            logger.info(f"Job {job.job_id}:")
            logger.info(f"  Status: {job.status.value}")
            logger.info(f"  Updated: {job.updated_time}")
            if job.progress_percentage:
                logger.info(f"  Progress: {job.progress_percentage:.1f}%")


async def demonstrate_job_cancellation():
    """Demonstrate job cancellation."""

    logger.info("Demonstrating job cancellation...")

    async with CloudTrainingAPI() as api:

        # Submit a job
        job_spec = TrainingJobSpec(
            job_name="cancellation-test",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_SMALL
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["file:///tmp/data"], output_path="file:///tmp/output"
            ),
            environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
        )

        result = await api.submit_job(CloudProvider.MOCK, job_spec)
        logger.info(f"Submitted job {result.job_id} for cancellation test")

        # Wait a moment for job to start
        await asyncio.sleep(2)

        # Cancel the job
        success = await api.cancel_job(CloudProvider.MOCK, result.job_id)

        if success:
            logger.info("Job cancelled successfully! ✅")
        else:
            logger.warning("Job could not be cancelled")

        # Check final status
        final_status = await api.get_job_status(CloudProvider.MOCK, result.job_id)
        logger.info(f"Final job status: {final_status.status.value}")


async def main():
    """Run all examples."""

    logger.info("=" * 60)
    logger.info("CloudTrain Basic Usage Examples")
    logger.info("=" * 60)

    try:
        # Example 1: Basic job submission
        logger.info("\n1. Basic Job Submission")
        logger.info("-" * 30)
        job_id = await basic_job_submission()

        # Example 2: Job monitoring
        logger.info("\n2. Job Monitoring")
        logger.info("-" * 30)
        async with CloudTrainingAPI() as api:
            await monitor_job_progress(api, job_id)

        # Example 3: Job listing
        logger.info("\n3. Job Listing")
        logger.info("-" * 30)
        await demonstrate_job_listing()

        # Example 4: Job cancellation
        logger.info("\n4. Job Cancellation")
        logger.info("-" * 30)
        await demonstrate_job_cancellation()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! 🎉")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
