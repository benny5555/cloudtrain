"""Unit tests for CloudTrain validation utilities."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from cloudtrain.enums import InstanceType
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    ResourceRequirements,
    TrainingJobSpec,
)
from cloudtrain.utils.validation import (
    ValidationError,
    _validate_data_configuration,
    _validate_data_path,
    _validate_environment_configuration,
    _validate_job_name,
    _validate_resource_requirements,
    _validate_tags,
    validate_job_spec,
    validate_provider_credentials,
)

# Test functions for validation functions (one-to-one mapping)


@pytest.mark.parametrize(
    "valid_name",
    [
        "test-job",
        "job123",
        "my-training-job-1",
        "a",
        "a" * 63,  # Maximum length
    ],
)
def test_validate_job_name_valid(valid_name):
    """Test _validate_job_name() function with valid names."""
    _validate_job_name(valid_name)  # Should not raise


@pytest.mark.parametrize(
    "invalid_name,expected_error",
    [
        ("", "Job name cannot be empty"),
        ("a" * 64, "Job name cannot exceed 63 characters"),
        ("job_with_underscore", "can only contain alphanumeric characters and hyphens"),
        ("job with spaces", "can only contain alphanumeric characters and hyphens"),
        ("-job", "cannot start or end with a hyphen"),
        ("job-", "cannot start or end with a hyphen"),
        ("job--name", "cannot contain consecutive hyphens"),
        ("job@name", "can only contain alphanumeric characters and hyphens"),
    ],
)
def test_validate_job_name_invalid(invalid_name, expected_error):
    """Test _validate_job_name() function with invalid names."""
    with pytest.raises(ValidationError) as exc_info:
        _validate_job_name(invalid_name)
    assert expected_error in str(exc_info.value)


def test_validate_resource_requirements_valid():
    """Test _validate_resource_requirements() function with valid requirements."""
    req = ResourceRequirements(
        instance_type=InstanceType.GPU_SMALL,
        instance_count=2,
        volume_size_gb=100,
        gpu_count=2,
        memory_gb=16.0,
        cpu_count=8,
    )
    _validate_resource_requirements(req)  # Should not raise


def test_validate_resource_requirements_custom_instance_type():
    """Test _validate_resource_requirements() function with custom instance type."""
    # Valid custom instance type
    req = ResourceRequirements(
        instance_type=InstanceType.CUSTOM, custom_instance_type="ml.p4d.24xlarge"
    )
    _validate_resource_requirements(req)  # Should not raise

    # Missing custom instance type - should fail at model creation
    with pytest.raises(PydanticValidationError) as exc_info:
        ResourceRequirements(instance_type=InstanceType.CUSTOM)
    assert "custom_instance_type is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_count,expected_error",
        [
            (101, "Instance count cannot exceed 100"),
            (200, "Instance count cannot exceed 100"),
        ],
    )
    def test_instance_count_validation_invalid(self, invalid_count, expected_error):
        """Test invalid instance count validation."""
        req = ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL, instance_count=invalid_count
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize("invalid_count", [0, -1])
    def test_instance_count_pydantic_validation_invalid(self, invalid_count):
        """Test invalid instance count caught by Pydantic validation."""
        with pytest.raises(PydanticValidationError):
            ResourceRequirements(
                instance_type=InstanceType.CPU_SMALL, instance_count=invalid_count
            )

    @pytest.mark.parametrize(
        "invalid_volume_size,expected_error",
        [
            (20000, "Volume size cannot exceed 16384 GB"),
            (50000, "Volume size cannot exceed 16384 GB"),
        ],
    )
    def test_volume_size_validation_invalid(self, invalid_volume_size, expected_error):
        """Test invalid volume size validation."""
        req = ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL, volume_size_gb=invalid_volume_size
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize("invalid_volume_size", [0, -1])
    def test_volume_size_pydantic_validation_invalid(self, invalid_volume_size):
        """Test invalid volume size caught by Pydantic validation."""
        with pytest.raises(PydanticValidationError):
            ResourceRequirements(
                instance_type=InstanceType.CPU_SMALL, volume_size_gb=invalid_volume_size
            )

    @pytest.mark.parametrize(
        "instance_type,gpu_count,expected_error",
        [
            (
                InstanceType.CPU_SMALL,
                1,
                "GPU count should only be specified for GPU instance types",
            ),
            (
                InstanceType.CPU_MEDIUM,
                2,
                "GPU count should only be specified for GPU instance types",
            ),
            (InstanceType.GPU_SMALL, 10, "GPU count cannot exceed 8"),
            (InstanceType.GPU_MEDIUM, 15, "GPU count cannot exceed 8"),
        ],
    )
    def test_gpu_count_validation_invalid(
        self, instance_type, gpu_count, expected_error
    ):
        """Test invalid GPU count validation."""
        req = ResourceRequirements(instance_type=instance_type, gpu_count=gpu_count)
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert expected_error in str(exc_info.value)


def test_validate_data_configuration_valid():
    """Test _validate_data_configuration() function with valid configuration."""
    config = DataConfiguration(
        input_data_paths=["s3://bucket/data1", "gs://bucket/data2"],
        output_path="s3://bucket/output",
        checkpoint_path="s3://bucket/checkpoints",
    )
    _validate_data_configuration(config)  # Should not raise


def test_validate_data_configuration_empty_input_paths():
    """Test _validate_data_configuration() function with empty input paths."""
    with pytest.raises(PydanticValidationError) as exc_info:
        DataConfiguration(input_data_paths=[], output_path="s3://bucket/output")
    assert "At least one input data path is required" in str(exc_info.value)


@pytest.mark.parametrize(
    "valid_path",
    [
        "s3://my-bucket/data/",
        "gs://my-bucket/data/",
        "gcs://my-bucket/data/",
        "azure://container/data/",
        "abfs://container/data/",
        "file:///tmp/data",
        "hdfs://namenode:9000/data",
    ],
)
def test_validate_data_path_valid(valid_path):
    """Test _validate_data_path() function with valid paths."""
    _validate_data_path(valid_path, "test")  # Should not raise


@pytest.mark.parametrize(
    "invalid_path,expected_error",
    [
        ("", "path cannot be empty"),
        ("ftp://server/data", "Unsupported test path scheme: ftp"),
        ("s3://", "Missing bucket name"),
        ("gs://", "Missing bucket name"),
        ("s3://invalid_bucket_name!", "Invalid bucket name"),
    ],
)
def test_validate_data_path_invalid(invalid_path, expected_error):
    """Test _validate_data_path() function with invalid paths."""
    with pytest.raises(ValidationError) as exc_info:
        _validate_data_path(invalid_path, "test")
    assert expected_error in str(exc_info.value)


def test_validate_environment_configuration_valid():
    """Test _validate_environment_configuration() function with valid configuration."""
    config = EnvironmentConfiguration(
        entry_point="train.py",
        python_version="3.9.0",
        framework_version="2.0.1",
        environment_variables={"CUDA_VISIBLE_DEVICES": "0,1", "BATCH_SIZE": "32"},
    )
    _validate_environment_configuration(config)  # Should not raise


@pytest.mark.parametrize(
    "invalid_entry_point,expected_error",
    [
        ("", "Entry point is required"),
        ("train.txt", "Entry point must be a Python (.py) or shell (.sh) script"),
    ],
)
def test_validate_environment_configuration_invalid_entry_point(
    invalid_entry_point, expected_error
):
    """Test _validate_environment_configuration() function with invalid entry points."""
    config = EnvironmentConfiguration(entry_point=invalid_entry_point)
    with pytest.raises(ValidationError) as exc_info:
        _validate_environment_configuration(config)
    assert expected_error in str(exc_info.value)

    def test_invalid_python_version(self):
        """Test invalid Python version validation."""
        config = EnvironmentConfiguration(
            entry_point="train.py", python_version="invalid"
        )

        with pytest.raises(ValidationError) as exc_info:
            _validate_environment_configuration(config)
        assert "Invalid Python version format" in str(exc_info.value)

    def test_invalid_environment_variables(self):
        """Test invalid environment variables validation."""
        config = EnvironmentConfiguration(
            entry_point="train.py", environment_variables={"": "value"}
        )

        with pytest.raises(ValidationError) as exc_info:
            _validate_environment_configuration(config)
        assert "Environment variable name cannot be empty" in str(exc_info.value)

        config = EnvironmentConfiguration(
            entry_point="train.py", environment_variables={"123invalid": "value"}
        )

        with pytest.raises(ValidationError) as exc_info:
            _validate_environment_configuration(config)
        assert "Invalid environment variable name" in str(exc_info.value)


def test_validate_tags_valid():
    """Test _validate_tags() function with valid tags."""
    valid_tags = {
        "project": "ml-training",
        "team": "data-science",
        "environment": "production",
        "cost-center": "12345",
    }
    _validate_tags(valid_tags)  # Should not raise


def test_validate_tags_empty():
    """Test _validate_tags() function with empty tags."""
    _validate_tags(None)  # Should not raise
    _validate_tags({})  # Should not raise

    def test_too_many_tags(self):
        """Test too many tags validation."""
        too_many_tags = {f"tag{i}": f"value{i}" for i in range(51)}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(too_many_tags)
        assert "Cannot have more than 50 tags" in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_tags,expected_error",
    [
        ({"": "value"}, "Tag key cannot be empty"),
        ({"a" * 129: "value"}, "Tag key cannot exceed 128 characters"),
    ],
)
def test_validate_tags_invalid_key(invalid_tags, expected_error):
    """Test _validate_tags() function with invalid tag keys."""
    with pytest.raises(ValidationError) as exc_info:
        _validate_tags(invalid_tags)
    assert expected_error in str(exc_info.value)


@pytest.mark.parametrize(
    "invalid_tags,expected_error",
    [
        ({"key": None}, "Tag value cannot be None"),
        ({"key": "a" * 257}, "Tag value cannot exceed 256 characters"),
    ],
)
def test_validate_tags_invalid_value(invalid_tags, expected_error):
    """Test _validate_tags() function with invalid tag values."""
    with pytest.raises(ValidationError) as exc_info:
        _validate_tags(invalid_tags)
    assert expected_error in str(exc_info.value)


def test_validate_job_spec_valid():
    """Test validate_job_spec() function with valid specification."""
    spec = TrainingJobSpec(
        job_name="test-job",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["s3://bucket/data"], output_path="s3://bucket/output"
        ),
        environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
        tags={"project": "test"},
    )
    validate_job_spec(spec)  # Should not raise

    def test_invalid_job_spec_propagates_errors(self):
        """Test that job spec validation propagates underlying errors."""
        # Invalid job spec should fail at creation time due to empty job_name
        with pytest.raises(PydanticValidationError) as exc_info:
            TrainingJobSpec(
                job_name="",  # Invalid
                resource_requirements=ResourceRequirements(
                    instance_type=InstanceType.GPU_SMALL
                ),
                data_configuration=DataConfiguration(
                    input_data_paths=["s3://bucket/data"],
                    output_path="s3://bucket/output",
                ),
                environment_configuration=EnvironmentConfiguration(
                    entry_point="train.py"
                ),
            )
        assert "String should have at least 1 character" in str(exc_info.value)


def test_validate_provider_credentials_valid():
    """Test validate_provider_credentials() function with valid configuration."""
    mock_config = type(
        "MockConfig", (), {"enabled": True, "is_valid": lambda self: True}
    )()
    errors = validate_provider_credentials(mock_config)
    assert errors == []


def test_validate_provider_credentials_missing():
    """Test validate_provider_credentials() function with missing configuration."""
    errors = validate_provider_credentials(None)
    assert "Provider configuration is missing" in errors


def test_validate_provider_credentials_disabled():
    """Test validate_provider_credentials() function with disabled configuration."""
    mock_config = type("MockConfig", (), {"enabled": False, "is_valid": lambda: True})()
    errors = validate_provider_credentials(mock_config)
    assert "Provider is disabled" in errors


def test_validate_provider_credentials_invalid():
    """Test validate_provider_credentials() function with invalid configuration."""
    mock_config = type(
        "MockConfig", (), {"enabled": True, "is_valid": lambda self: False}
    )()
    errors = validate_provider_credentials(mock_config)
    assert "Provider configuration is incomplete or invalid" in errors
