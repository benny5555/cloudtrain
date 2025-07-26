"""Validation utilities for CloudTrain.

This module provides validation functions for job specifications,
configurations, and other data structures used in CloudTrain.
"""

import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from cloudtrain.enums import InstanceType
from cloudtrain.schemas import TrainingJobSpec


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass


def validate_job_spec(job_spec: TrainingJobSpec) -> None:
    """Validate a training job specification.

    This function performs comprehensive validation of a training job
    specification to ensure it meets all requirements and constraints.

    Args:
        job_spec: Training job specification to validate

    Raises:
        ValidationError: If validation fails
    """
    # Validate job name
    _validate_job_name(job_spec.job_name)

    # Validate resource requirements
    _validate_resource_requirements(job_spec.resource_requirements)

    # Validate data configuration
    _validate_data_configuration(job_spec.data_configuration)

    # Validate environment configuration
    _validate_environment_configuration(job_spec.environment_configuration)

    # Validate tags
    _validate_tags(job_spec.tags)


def _validate_job_name(job_name: str) -> None:
    """Validate job name format and constraints.

    Args:
        job_name: Job name to validate

    Raises:
        ValidationError: If job name is invalid
    """
    if not job_name:
        raise ValidationError("Job name cannot be empty")

    if len(job_name) > 63:
        raise ValidationError("Job name cannot exceed 63 characters")

    # Check for valid characters (alphanumeric and hyphens)
    if not re.match(r"^[a-zA-Z0-9\-]+$", job_name):
        raise ValidationError(
            "Job name can only contain alphanumeric characters and hyphens"
        )

    # Cannot start or end with hyphen
    if job_name.startswith("-") or job_name.endswith("-"):
        raise ValidationError("Job name cannot start or end with a hyphen")

    # Cannot have consecutive hyphens
    if "--" in job_name:
        raise ValidationError("Job name cannot contain consecutive hyphens")


def _validate_resource_requirements(resource_req) -> None:
    """Validate resource requirements.

    Args:
        resource_req: Resource requirements to validate

    Raises:
        ValidationError: If resource requirements are invalid
    """
    # Validate instance type and custom type combination
    if resource_req.instance_type == InstanceType.CUSTOM:
        if not resource_req.custom_instance_type:
            raise ValidationError(
                "custom_instance_type is required when instance_type is CUSTOM"
            )
    else:
        if resource_req.custom_instance_type:
            raise ValidationError(
                "custom_instance_type should only be set when instance_type is CUSTOM"
            )

    # Validate instance count
    if resource_req.instance_count < 1:
        raise ValidationError("Instance count must be at least 1")

    if resource_req.instance_count > 100:
        raise ValidationError("Instance count cannot exceed 100")

    # Validate volume size
    if resource_req.volume_size_gb < 1:
        raise ValidationError("Volume size must be at least 1 GB")

    if resource_req.volume_size_gb > 16384:  # 16 TB
        raise ValidationError("Volume size cannot exceed 16384 GB")

    # Validate GPU count if specified
    if resource_req.gpu_count is not None:
        if resource_req.gpu_count < 1:
            raise ValidationError("GPU count must be at least 1 if specified")

        if resource_req.gpu_count > 8:
            raise ValidationError("GPU count cannot exceed 8 per instance")

        # GPU count should only be specified for GPU instance types
        if (
            not resource_req.instance_type.has_gpu()
            and resource_req.instance_type != InstanceType.CUSTOM
        ):
            raise ValidationError(
                "GPU count should only be specified for GPU instance types"
            )

    # Validate memory if specified
    if resource_req.memory_gb is not None:
        if resource_req.memory_gb < 0.5:
            raise ValidationError("Memory must be at least 0.5 GB if specified")

        if resource_req.memory_gb > 1024:  # 1 TB
            raise ValidationError("Memory cannot exceed 1024 GB")

    # Validate CPU count if specified
    if resource_req.cpu_count is not None:
        if resource_req.cpu_count < 1:
            raise ValidationError("CPU count must be at least 1 if specified")

        if resource_req.cpu_count > 128:
            raise ValidationError("CPU count cannot exceed 128")


def _validate_data_configuration(data_config) -> None:
    """Validate data configuration.

    Args:
        data_config: Data configuration to validate

    Raises:
        ValidationError: If data configuration is invalid
    """
    # Validate input data paths
    if not data_config.input_data_paths:
        raise ValidationError("At least one input data path is required")

    for path in data_config.input_data_paths:
        _validate_data_path(path, "input")

    # Validate output path
    _validate_data_path(data_config.output_path, "output")

    # Validate checkpoint path if specified
    if data_config.checkpoint_path:
        _validate_data_path(data_config.checkpoint_path, "checkpoint")


def _validate_data_path(path: str, path_type: str) -> None:
    """Validate a data path (S3, GCS, Azure Blob, etc.).

    Args:
        path: Data path to validate
        path_type: Type of path for error messages

    Raises:
        ValidationError: If path is invalid
    """
    if not path:
        raise ValidationError(f"{path_type.title()} path cannot be empty")

    # Parse URL to validate format
    try:
        parsed = urlparse(path)
    except Exception:
        raise ValidationError(f"Invalid {path_type} path format: {path}")

    # Check for supported schemes
    supported_schemes = {"s3", "gs", "gcs", "azure", "abfs", "abfss", "file", "hdfs"}

    if parsed.scheme and parsed.scheme.lower() not in supported_schemes:
        raise ValidationError(
            f"Unsupported {path_type} path scheme: {parsed.scheme}. "
            f"Supported schemes: {', '.join(sorted(supported_schemes))}"
        )

    # Validate bucket/container name for cloud storage
    if parsed.scheme in {"s3", "gs", "gcs"}:
        if not parsed.netloc:
            raise ValidationError(f"Missing bucket name in {path_type} path: {path}")

        # Basic bucket name validation
        bucket_name = parsed.netloc
        if not re.match(r"^[a-z0-9][a-z0-9\-\.]*[a-z0-9]$", bucket_name):
            raise ValidationError(
                f"Invalid bucket name in {path_type} path: {bucket_name}"
            )


def _validate_environment_configuration(env_config) -> None:
    """Validate environment configuration.

    Args:
        env_config: Environment configuration to validate

    Raises:
        ValidationError: If environment configuration is invalid
    """
    # Validate entry point
    if not env_config.entry_point:
        raise ValidationError("Entry point is required")

    # Validate entry point format
    if not env_config.entry_point.endswith((".py", ".sh")):
        raise ValidationError(
            "Entry point must be a Python (.py) or shell (.sh) script"
        )

    # Validate Python version if specified
    if env_config.python_version:
        if not re.match(r"^\d+\.\d+(\.\d+)?$", env_config.python_version):
            raise ValidationError(
                "Invalid Python version format (expected: X.Y or X.Y.Z)"
            )

    # Validate framework version if specified
    if env_config.framework_version:
        if not re.match(r"^\d+\.\d+(\.\d+)?", env_config.framework_version):
            raise ValidationError("Invalid framework version format")

    # Validate environment variables
    if env_config.environment_variables:
        for key, value in env_config.environment_variables.items():
            if not key:
                raise ValidationError("Environment variable name cannot be empty")

            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValidationError(f"Invalid environment variable name: {key}")

            if value is None:
                raise ValidationError(
                    f"Environment variable value cannot be None: {key}"
                )


def _validate_tags(tags: Optional[dict]) -> None:
    """Validate job tags.

    Args:
        tags: Tags dictionary to validate

    Raises:
        ValidationError: If tags are invalid
    """
    if not tags:
        return

    if len(tags) > 50:
        raise ValidationError("Cannot have more than 50 tags")

    for key, value in tags.items():
        # Validate tag key
        if not key:
            raise ValidationError("Tag key cannot be empty")

        if len(key) > 128:
            raise ValidationError("Tag key cannot exceed 128 characters")

        if not re.match(r"^[a-zA-Z0-9\-_\.:/]+$", key):
            raise ValidationError(f"Invalid tag key format: {key}")

        # Validate tag value
        if value is None:
            raise ValidationError(f"Tag value cannot be None for key: {key}")

        if len(str(value)) > 256:
            raise ValidationError(
                f"Tag value cannot exceed 256 characters for key: {key}"
            )


def validate_provider_credentials(provider_config) -> List[str]:
    """Validate provider credentials and configuration.

    Args:
        provider_config: Provider configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not provider_config:
        errors.append("Provider configuration is missing")
        return errors

    if not provider_config.enabled:
        errors.append("Provider is disabled")
        return errors

    if not provider_config.is_valid():
        errors.append("Provider configuration is incomplete or invalid")

    return errors
