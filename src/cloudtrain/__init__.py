"""CloudTrain: Universal Cloud Training API.

A Python library for submitting machine learning training jobs to multiple cloud providers
through a unified interface.

Example:
    Basic usage:

    >>> from cloudtrain import CloudTrainingAPI, CloudProvider, TrainingJobSpec
    >>> api = CloudTrainingAPI()
    >>> job_spec = TrainingJobSpec(
    ...     job_name="my-training-job",
    ...     script_path="train.py",
    ...     instance_type="ml.m5.large",
    ...     output_path="s3://my-bucket/output/"
    ... )
    >>> result = await api.submit_job(CloudProvider.AWS, job_spec)
    >>> print(f"Job submitted: {result.job_id}")

Attributes:
    __version__: The version of the CloudTrain package.
"""

from cloudtrain.api import CloudTrainingAPI
from cloudtrain.enums import CloudProvider, InstanceType, JobStatus, LogLevel
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)

__version__ = "0.1.0"
__author__ = "CloudTrain Team"
__email__ = "team@cloudtrain.ai"

__all__ = [
    "CloudTrainingAPI",
    "CloudProvider",
    "DataConfiguration",
    "EnvironmentConfiguration",
    "InstanceType",
    "JobStatus",
    "JobStatusUpdate",
    "LogLevel",
    "ResourceRequirements",
    "TrainingJobResult",
    "TrainingJobSpec",
    "__version__",
]
