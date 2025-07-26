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


class TestValidateJobName:
    """Test job name validation."""

    def test_valid_job_names(self):
        """Test valid job name validation."""
        valid_names = [
            "test-job",
            "job123",
            "my-training-job-1",
            "a",
            "a" * 63,  # Maximum length
        ]

        for name in valid_names:
            _validate_job_name(name)  # Should not raise

    def test_invalid_job_names(self):
        """Test invalid job name validation."""
        invalid_cases = [
            ("", "Job name cannot be empty"),
            ("a" * 64, "Job name cannot exceed 63 characters"),
            (
                "job_with_underscore",
                "can only contain alphanumeric characters and hyphens",
            ),
            ("job with spaces", "can only contain alphanumeric characters and hyphens"),
            ("-job", "cannot start or end with a hyphen"),
            ("job-", "cannot start or end with a hyphen"),
            ("job--name", "cannot contain consecutive hyphens"),
            ("job@name", "can only contain alphanumeric characters and hyphens"),
        ]

        for name, expected_error in invalid_cases:
            with pytest.raises(ValidationError) as exc_info:
                _validate_job_name(name)
            assert expected_error in str(exc_info.value)


class TestValidateResourceRequirements:
    """Test resource requirements validation."""

    def test_valid_resource_requirements(self):
        """Test valid resource requirements validation."""
        req = ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL,
            instance_count=2,
            volume_size_gb=100,
            gpu_count=2,
            memory_gb=16.0,
            cpu_count=8,
        )

        _validate_resource_requirements(req)  # Should not raise

    def test_custom_instance_type_validation(self):
        """Test custom instance type validation."""
        # Valid custom instance type
        req = ResourceRequirements(
            instance_type=InstanceType.CUSTOM, custom_instance_type="ml.p4d.24xlarge"
        )
        _validate_resource_requirements(req)  # Should not raise

        # Missing custom instance type - should fail at model creation
        with pytest.raises(PydanticValidationError) as exc_info:
            ResourceRequirements(instance_type=InstanceType.CUSTOM)
        assert "custom_instance_type is required" in str(exc_info.value)

    def test_instance_count_validation(self):
        """Test instance count validation."""
        # Invalid instance count
        req = ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL, instance_count=101
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert "Instance count cannot exceed 100" in str(exc_info.value)

    def test_volume_size_validation(self):
        """Test volume size validation."""
        # Invalid volume size
        req = ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL, volume_size_gb=20000
        )
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert "Volume size cannot exceed 16384 GB" in str(exc_info.value)

    def test_gpu_count_validation(self):
        """Test GPU count validation."""
        # GPU count with CPU instance type
        req = ResourceRequirements(instance_type=InstanceType.CPU_SMALL, gpu_count=2)
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert "GPU count should only be specified for GPU instance types" in str(
            exc_info.value
        )

        # Invalid GPU count
        req = ResourceRequirements(instance_type=InstanceType.GPU_SMALL, gpu_count=10)
        with pytest.raises(ValidationError) as exc_info:
            _validate_resource_requirements(req)
        assert "GPU count cannot exceed 8" in str(exc_info.value)


class TestValidateDataConfiguration:
    """Test data configuration validation."""

    def test_valid_data_configuration(self):
        """Test valid data configuration validation."""
        config = DataConfiguration(
            input_data_paths=["s3://bucket/data1", "gs://bucket/data2"],
            output_path="s3://bucket/output",
            checkpoint_path="s3://bucket/checkpoints",
        )

        _validate_data_configuration(config)  # Should not raise

    def test_empty_input_paths(self):
        """Test empty input paths validation."""
        with pytest.raises(PydanticValidationError) as exc_info:
            DataConfiguration(input_data_paths=[], output_path="s3://bucket/output")
        assert "At least one input data path is required" in str(exc_info.value)


class TestValidateDataPath:
    """Test data path validation."""

    def test_valid_data_paths(self):
        """Test valid data path validation."""
        valid_paths = [
            "s3://my-bucket/data/",
            "gs://my-bucket/data/",
            "gcs://my-bucket/data/",
            "azure://container/data/",
            "abfs://container/data/",
            "file:///tmp/data",
            "hdfs://namenode:9000/data",
        ]

        for path in valid_paths:
            _validate_data_path(path, "test")  # Should not raise

    def test_invalid_data_paths(self):
        """Test invalid data path validation."""
        invalid_cases = [
            ("", "path cannot be empty"),
            ("ftp://server/data", "Unsupported test path scheme: ftp"),
            ("s3://", "Missing bucket name"),
            ("gs://", "Missing bucket name"),
            ("s3://invalid_bucket_name!", "Invalid bucket name"),
        ]

        for path, expected_error in invalid_cases:
            with pytest.raises(ValidationError) as exc_info:
                _validate_data_path(path, "test")
            assert expected_error in str(exc_info.value)


class TestValidateEnvironmentConfiguration:
    """Test environment configuration validation."""

    def test_valid_environment_configuration(self):
        """Test valid environment configuration validation."""
        config = EnvironmentConfiguration(
            entry_point="train.py",
            python_version="3.9.0",
            framework_version="2.0.1",
            environment_variables={"CUDA_VISIBLE_DEVICES": "0,1", "BATCH_SIZE": "32"},
        )

        _validate_environment_configuration(config)  # Should not raise

    def test_invalid_entry_point(self):
        """Test invalid entry point validation."""
        config = EnvironmentConfiguration(entry_point="")

        with pytest.raises(ValidationError) as exc_info:
            _validate_environment_configuration(config)
        assert "Entry point is required" in str(exc_info.value)

        config = EnvironmentConfiguration(entry_point="train.txt")

        with pytest.raises(ValidationError) as exc_info:
            _validate_environment_configuration(config)
        assert "Entry point must be a Python (.py) or shell (.sh) script" in str(
            exc_info.value
        )

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


class TestValidateTags:
    """Test tags validation."""

    def test_valid_tags(self):
        """Test valid tags validation."""
        valid_tags = {
            "project": "ml-training",
            "team": "data-science",
            "environment": "production",
            "cost-center": "12345",
        }

        _validate_tags(valid_tags)  # Should not raise

    def test_empty_tags(self):
        """Test empty tags validation."""
        _validate_tags(None)  # Should not raise
        _validate_tags({})  # Should not raise

    def test_too_many_tags(self):
        """Test too many tags validation."""
        too_many_tags = {f"tag{i}": f"value{i}" for i in range(51)}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(too_many_tags)
        assert "Cannot have more than 50 tags" in str(exc_info.value)

    def test_invalid_tag_key(self):
        """Test invalid tag key validation."""
        invalid_tags = {"": "value"}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(invalid_tags)
        assert "Tag key cannot be empty" in str(exc_info.value)

        invalid_tags = {"a" * 129: "value"}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(invalid_tags)
        assert "Tag key cannot exceed 128 characters" in str(exc_info.value)

    def test_invalid_tag_value(self):
        """Test invalid tag value validation."""
        invalid_tags = {"key": None}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(invalid_tags)
        assert "Tag value cannot be None" in str(exc_info.value)

        invalid_tags = {"key": "a" * 257}

        with pytest.raises(ValidationError) as exc_info:
            _validate_tags(invalid_tags)
        assert "Tag value cannot exceed 256 characters" in str(exc_info.value)


class TestValidateJobSpec:
    """Test complete job specification validation."""

    def test_valid_job_spec(self):
        """Test valid job specification validation."""
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


class TestValidateProviderCredentials:
    """Test provider credentials validation."""

    def test_valid_provider_config(self):
        """Test valid provider configuration validation."""
        mock_config = type(
            "MockConfig", (), {"enabled": True, "is_valid": lambda self: True}
        )()

        errors = validate_provider_credentials(mock_config)
        assert errors == []

    def test_missing_provider_config(self):
        """Test missing provider configuration validation."""
        errors = validate_provider_credentials(None)
        assert "Provider configuration is missing" in errors

    def test_disabled_provider_config(self):
        """Test disabled provider configuration validation."""
        mock_config = type(
            "MockConfig", (), {"enabled": False, "is_valid": lambda: True}
        )()

        errors = validate_provider_credentials(mock_config)
        assert "Provider is disabled" in errors

    def test_invalid_provider_config(self):
        """Test invalid provider configuration validation."""
        mock_config = type(
            "MockConfig", (), {"enabled": True, "is_valid": lambda self: False}
        )()

        errors = validate_provider_credentials(mock_config)
        assert "Provider configuration is incomplete or invalid" in errors
