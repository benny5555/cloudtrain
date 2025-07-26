"""Data schemas and models for CloudTrain API.

This module defines the Pydantic models used for data validation,
serialization, and API contracts in the CloudTrain universal
cloud training API.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus, LogLevel


class ResourceRequirements(BaseModel):
    """Resource requirements for training jobs.

    This model defines the compute resources needed for a training job,
    including instance specifications, scaling parameters, and resource limits.

    Attributes:
        instance_type: Type of compute instance to use
        instance_count: Number of instances for distributed training
        custom_instance_type: Provider-specific instance type (if instance_type is CUSTOM)
        max_runtime_seconds: Maximum allowed runtime in seconds
        volume_size_gb: Storage volume size in GB
        gpu_count: Number of GPUs per instance (if applicable)
        memory_gb: Memory requirement in GB (optional override)
        cpu_count: CPU count requirement (optional override)
    """

    instance_type: InstanceType = Field(
        description="Standardized instance type for the training job"
    )
    instance_count: PositiveInt = Field(
        default=1, description="Number of instances for distributed training"
    )
    custom_instance_type: Optional[str] = Field(
        default=None, description="Provider-specific instance type when using CUSTOM"
    )
    max_runtime_seconds: Optional[PositiveInt] = Field(
        default=None, description="Maximum runtime in seconds (None for unlimited)"
    )
    volume_size_gb: PositiveInt = Field(
        default=30, description="Storage volume size in GB"
    )
    gpu_count: Optional[PositiveInt] = Field(
        default=None, description="Number of GPUs per instance"
    )
    memory_gb: Optional[PositiveFloat] = Field(
        default=None, description="Memory requirement in GB"
    )
    cpu_count: Optional[PositiveInt] = Field(
        default=None, description="CPU count requirement"
    )

    @model_validator(mode="after")
    def validate_custom_instance_type(self) -> "ResourceRequirements":
        """Validate custom instance type is provided when needed."""
        if self.instance_type == InstanceType.CUSTOM and not self.custom_instance_type:
            raise ValueError(
                "custom_instance_type is required when instance_type is CUSTOM"
            )
        if self.instance_type != InstanceType.CUSTOM and self.custom_instance_type:
            raise ValueError(
                "custom_instance_type should only be set when instance_type is CUSTOM"
            )
        return self


class DataConfiguration(BaseModel):
    """Data configuration for training jobs.

    This model defines the input and output data configuration
    for training jobs, including data sources, preprocessing,
    and output destinations.

    Attributes:
        input_data_paths: List of input data paths (S3, GCS, Azure Blob, etc.)
        output_path: Output path for training artifacts
        checkpoint_path: Path for saving training checkpoints
        data_preprocessing: Optional preprocessing configuration
        data_format: Format of the input data (csv, json, parquet, etc.)
    """

    input_data_paths: List[str] = Field(description="List of input data paths")
    output_path: str = Field(description="Output path for training artifacts")
    checkpoint_path: Optional[str] = Field(
        default=None, description="Path for saving training checkpoints"
    )
    data_preprocessing: Optional[Dict[str, Any]] = Field(
        default=None, description="Preprocessing configuration"
    )
    data_format: Optional[str] = Field(default=None, description="Format of input data")

    @field_validator("input_data_paths")
    @classmethod
    def validate_input_paths(cls, v: List[str]) -> List[str]:
        """Validate input data paths are not empty."""
        if not v:
            raise ValueError("At least one input data path is required")
        return v


class EnvironmentConfiguration(BaseModel):
    """Environment configuration for training jobs.

    This model defines the runtime environment for training jobs,
    including container images, environment variables, and dependencies.

    Attributes:
        container_image: Docker container image for training
        python_version: Python version requirement
        framework: ML framework (tensorflow, pytorch, sklearn, etc.)
        framework_version: Version of the ML framework
        environment_variables: Environment variables for the job
        requirements_file: Path to requirements.txt or similar
        entry_point: Entry point script for training
        command_line_args: Command line arguments for the entry point
    """

    container_image: Optional[str] = Field(
        default=None, description="Docker container image for training"
    )
    python_version: Optional[str] = Field(
        default=None, description="Python version requirement"
    )
    framework: Optional[str] = Field(default=None, description="ML framework name")
    framework_version: Optional[str] = Field(
        default=None, description="ML framework version"
    )
    environment_variables: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Environment variables for the job"
    )
    requirements_file: Optional[str] = Field(
        default=None, description="Path to requirements file"
    )
    entry_point: str = Field(description="Entry point script for training")
    command_line_args: Optional[List[str]] = Field(
        default_factory=list, description="Command line arguments"
    )


class TrainingJobSpec(BaseModel):
    """Complete specification for a training job.

    This is the main model that defines all parameters needed
    to submit a training job to any supported cloud provider.

    Attributes:
        job_name: Unique name for the training job
        description: Optional description of the job
        resource_requirements: Compute resource specifications
        data_configuration: Data input/output configuration
        environment_configuration: Runtime environment setup
        tags: Optional tags for job organization
        provider_specific_config: Provider-specific configuration overrides
    """

    job_name: str = Field(
        description="Unique name for the training job",
        min_length=1,
        max_length=63,
        pattern=r"^[a-zA-Z0-9\-]+$",
    )
    description: Optional[str] = Field(
        default=None, description="Optional description of the job", max_length=1024
    )
    resource_requirements: ResourceRequirements = Field(
        description="Compute resource specifications"
    )
    data_configuration: DataConfiguration = Field(
        description="Data input/output configuration"
    )
    environment_configuration: EnvironmentConfiguration = Field(
        description="Runtime environment setup"
    )
    tags: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Tags for job organization"
    )
    provider_specific_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Provider-specific configuration overrides"
    )

    model_config = ConfigDict(
        use_enum_values=False, validate_assignment=True, extra="forbid"
    )


class TrainingJobResult(BaseModel):
    """Result of a training job submission.

    This model represents the response from submitting a training job,
    including job identification, status, and metadata.

    Attributes:
        job_id: Unique identifier assigned by the cloud provider
        job_name: Name of the submitted job
        provider: Cloud provider where the job was submitted
        status: Current status of the job
        submission_time: When the job was submitted
        estimated_start_time: Estimated time when job will start
        estimated_cost: Estimated cost for the job (if available)
        provider_job_url: URL to view job in provider console
        metadata: Additional metadata from the provider
    """

    job_id: str = Field(description="Unique identifier assigned by the cloud provider")
    job_name: str = Field(description="Name of the submitted job")
    provider: CloudProvider = Field(
        description="Cloud provider where the job was submitted"
    )
    status: JobStatus = Field(description="Current status of the job")
    submission_time: datetime = Field(description="When the job was submitted")
    estimated_start_time: Optional[datetime] = Field(
        default=None, description="Estimated time when job will start"
    )
    estimated_cost: Optional[float] = Field(
        default=None, description="Estimated cost for the job in USD"
    )
    provider_job_url: Optional[str] = Field(
        default=None, description="URL to view job in provider console"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata from the provider"
    )

    model_config = ConfigDict(use_enum_values=False)


class JobStatusUpdate(BaseModel):
    """Status update for a training job.

    This model represents a status update for a training job,
    including current status, progress information, and logs.

    Attributes:
        job_id: Unique identifier of the job
        status: Current status of the job
        progress_percentage: Training progress as percentage (0-100)
        current_epoch: Current training epoch (if applicable)
        total_epochs: Total number of epochs (if applicable)
        metrics: Current training metrics
        logs: Recent log messages
        error_message: Error message if job failed
        updated_time: When this status was last updated
    """

    job_id: str = Field(description="Unique identifier of the job")
    status: JobStatus = Field(description="Current status of the job")
    progress_percentage: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Training progress as percentage"
    )
    current_epoch: Optional[int] = Field(
        default=None, ge=0, description="Current training epoch"
    )
    total_epochs: Optional[int] = Field(
        default=None, ge=1, description="Total number of epochs"
    )
    metrics: Optional[Dict[str, float]] = Field(
        default_factory=dict, description="Current training metrics"
    )
    logs: Optional[List[str]] = Field(
        default_factory=list, description="Recent log messages"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )
    updated_time: datetime = Field(description="When this status was last updated")

    model_config = ConfigDict(use_enum_values=False)
