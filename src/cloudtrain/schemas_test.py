"""Unit tests for CloudTrain schemas."""

from datetime import UTC, datetime
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from cloudtrain.enums import CloudProvider, InstanceType, JobStatus
from cloudtrain.schemas import (
    DataConfiguration,
    EnvironmentConfiguration,
    JobStatusUpdate,
    ResourceRequirements,
    TrainingJobResult,
    TrainingJobSpec,
)


class TestResourceRequirements:
    """Test ResourceRequirements schema."""

    def test_valid_resource_requirements(self) -> None:
        """Test creating valid resource requirements."""
        req: ResourceRequirements = ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL,
            instance_count=2,
            volume_size_gb=100,
            max_runtime_seconds=3600,
        )

        assert req.instance_type == InstanceType.GPU_SMALL
        assert req.instance_count == 2
        assert req.volume_size_gb == 100
        assert req.max_runtime_seconds == 3600

    def test_default_values(self) -> None:
        """Test default values for resource requirements."""
        req: ResourceRequirements = ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL
        )

        assert req.instance_count == 1
        assert req.volume_size_gb == 30
        assert req.custom_instance_type is None
        assert req.max_runtime_seconds is None

    def test_custom_instance_type_validation_valid(self) -> None:
        """Test valid custom instance type validation."""
        req: ResourceRequirements = ResourceRequirements(
            instance_type=InstanceType.CUSTOM, custom_instance_type="ml.p4d.24xlarge"
        )
        assert req.custom_instance_type == "ml.p4d.24xlarge"

    def test_custom_instance_type_validation_missing(self):
        """Test custom instance type validation when missing."""
        with pytest.raises(ValidationError) as exc_info:
            ResourceRequirements(instance_type=InstanceType.CUSTOM)
        assert "custom_instance_type is required" in str(exc_info.value)

    def test_custom_instance_type_validation_invalid_usage(self):
        """Test custom instance type validation when used incorrectly."""
        with pytest.raises(ValidationError) as exc_info:
            ResourceRequirements(
                instance_type=InstanceType.GPU_SMALL,
                custom_instance_type="ml.p4d.24xlarge",
            )
        assert "should only be set when instance_type is CUSTOM" in str(exc_info.value)


class TestDataConfiguration:
    """Test DataConfiguration schema."""

    def test_valid_data_configuration(self):
        """Test creating valid data configuration."""
        config = DataConfiguration(
            input_data_paths=["s3://bucket/data1", "s3://bucket/data2"],
            output_path="s3://bucket/output",
            checkpoint_path="s3://bucket/checkpoints",
            data_format="parquet",
        )

        assert len(config.input_data_paths) == 2
        assert config.output_path == "s3://bucket/output"
        assert config.checkpoint_path == "s3://bucket/checkpoints"
        assert config.data_format == "parquet"

    def test_empty_input_paths_validation(self):
        """Test validation of empty input paths."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfiguration(input_data_paths=[], output_path="s3://bucket/output")
        assert "At least one input data path is required" in str(exc_info.value)

    def test_optional_fields(self):
        """Test optional fields in data configuration."""
        config = DataConfiguration(
            input_data_paths=["s3://bucket/data"], output_path="s3://bucket/output"
        )

        assert config.checkpoint_path is None
        assert config.data_preprocessing is None
        assert config.data_format is None


class TestEnvironmentConfiguration:
    """Test EnvironmentConfiguration schema."""

    def test_valid_environment_configuration(self):
        """Test creating valid environment configuration."""
        config = EnvironmentConfiguration(
            entry_point="train.py",
            framework="pytorch",
            framework_version="2.0.0",
            python_version="3.9",
            environment_variables={"CUDA_VISIBLE_DEVICES": "0,1"},
            command_line_args=["--epochs", "10"],
        )

        assert config.entry_point == "train.py"
        assert config.framework == "pytorch"
        assert config.framework_version == "2.0.0"
        assert config.python_version == "3.9"
        assert config.environment_variables["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert config.command_line_args == ["--epochs", "10"]

    def test_default_values(self):
        """Test default values for environment configuration."""
        config = EnvironmentConfiguration(entry_point="train.py")

        assert config.container_image is None
        assert config.python_version is None
        assert config.framework is None
        assert config.framework_version is None
        assert config.environment_variables == {}
        assert config.requirements_file is None
        assert config.command_line_args == []


class TestTrainingJobSpec:
    """Test TrainingJobSpec schema."""

    def test_valid_training_job_spec(self):
        """Test creating valid training job specification."""
        spec = TrainingJobSpec(
            job_name="test-job",
            description="Test training job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.GPU_SMALL
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["s3://bucket/data"], output_path="s3://bucket/output"
            ),
            environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
            tags={"project": "test", "team": "ml"},
        )

        assert spec.job_name == "test-job"
        assert spec.description == "Test training job"
        assert spec.tags["project"] == "test"

    @pytest.mark.parametrize(
        "valid_name", ["test-job", "job123", "my-training-job-1", "a", "a-b-c-d-e-f"]
    )
    def test_job_name_validation_valid(self, valid_name):
        """Test valid job name validation."""
        spec = TrainingJobSpec(
            job_name=valid_name,
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_SMALL
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["s3://bucket/data"],
                output_path="s3://bucket/output",
            ),
            environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
        )
        assert spec.job_name == valid_name

    @pytest.mark.parametrize(
        "invalid_name", ["", "job_with_underscore", "job with spaces", "a" * 64]
    )
    def test_job_name_validation_invalid(self, invalid_name):
        """Test invalid job name validation."""
        with pytest.raises(ValidationError):
            TrainingJobSpec(
                job_name=invalid_name,
                resource_requirements=ResourceRequirements(
                    instance_type=InstanceType.CPU_SMALL
                ),
                data_configuration=DataConfiguration(
                    input_data_paths=["s3://bucket/data"],
                    output_path="s3://bucket/output",
                ),
                environment_configuration=EnvironmentConfiguration(
                    entry_point="train.py"
                ),
            )

    def test_default_values(self):
        """Test default values for training job spec."""
        spec = TrainingJobSpec(
            job_name="test-job",
            resource_requirements=ResourceRequirements(
                instance_type=InstanceType.CPU_SMALL
            ),
            data_configuration=DataConfiguration(
                input_data_paths=["s3://bucket/data"], output_path="s3://bucket/output"
            ),
            environment_configuration=EnvironmentConfiguration(entry_point="train.py"),
        )

        assert spec.description is None
        assert spec.tags == {}
        assert spec.provider_specific_config == {}


class TestTrainingJobResult:
    """Test TrainingJobResult schema."""

    def test_valid_training_job_result(self):
        """Test creating valid training job result."""
        result = TrainingJobResult(
            job_id="job-12345",
            job_name="test-job",
            provider=CloudProvider.AWS,
            status=JobStatus.PENDING,
            submission_time=datetime.now(UTC),
            estimated_cost=25.50,
            provider_job_url="https://console.aws.amazon.com/sagemaker/home#/jobs/job-12345",
        )

        assert result.job_id == "job-12345"
        assert result.job_name == "test-job"
        assert result.provider == CloudProvider.AWS
        assert result.status == JobStatus.PENDING
        assert result.estimated_cost == 25.50
        assert "sagemaker" in result.provider_job_url

    def test_default_values(self):
        """Test default values for training job result."""
        result = TrainingJobResult(
            job_id="job-12345",
            job_name="test-job",
            provider=CloudProvider.MOCK,
            status=JobStatus.PENDING,
            submission_time=datetime.now(UTC),
        )

        assert result.estimated_start_time is None
        assert result.estimated_cost is None
        assert result.provider_job_url is None
        assert result.metadata == {}


class TestJobStatusUpdate:
    """Test JobStatusUpdate schema."""

    def test_valid_job_status_update(self):
        """Test creating valid job status update."""
        update = JobStatusUpdate(
            job_id="job-12345",
            status=JobStatus.RUNNING,
            progress_percentage=75.5,
            current_epoch=8,
            total_epochs=10,
            metrics={"loss": 0.25, "accuracy": 0.92},
            logs=["Training started", "Epoch 8 completed"],
            updated_time=datetime.now(UTC),
        )

        assert update.job_id == "job-12345"
        assert update.status == JobStatus.RUNNING
        assert update.progress_percentage == 75.5
        assert update.current_epoch == 8
        assert update.total_epochs == 10
        assert update.metrics["loss"] == 0.25
        assert len(update.logs) == 2

    @pytest.mark.parametrize("valid_progress", [0.0, 25.5, 50.0, 75.5, 100.0])
    def test_progress_percentage_validation_valid(self, valid_progress):
        """Test valid progress percentage validation."""
        update = JobStatusUpdate(
            job_id="job-12345",
            status=JobStatus.RUNNING,
            progress_percentage=valid_progress,
            updated_time=datetime.now(UTC),
        )
        assert update.progress_percentage == valid_progress

    @pytest.mark.parametrize("invalid_progress", [-1.0, 101.0, 150.0, -10.5])
    def test_progress_percentage_validation_invalid(self, invalid_progress):
        """Test invalid progress percentage validation."""
        with pytest.raises(ValidationError):
            JobStatusUpdate(
                job_id="job-12345",
                status=JobStatus.RUNNING,
                progress_percentage=invalid_progress,
                updated_time=datetime.now(UTC),
            )

    def test_epoch_validation_valid(self):
        """Test valid epoch validation."""
        update = JobStatusUpdate(
            job_id="job-12345",
            status=JobStatus.RUNNING,
            current_epoch=5,
            total_epochs=10,
            updated_time=datetime.now(UTC),
        )
        assert update.current_epoch == 5
        assert update.total_epochs == 10

    def test_epoch_validation_invalid_current(self):
        """Test invalid current epoch validation."""
        with pytest.raises(ValidationError):
            JobStatusUpdate(
                job_id="job-12345",
                status=JobStatus.RUNNING,
                current_epoch=-1,
                updated_time=datetime.now(UTC),
            )

    def test_epoch_validation_invalid_total(self):
        """Test invalid total epochs validation."""
        with pytest.raises(ValidationError):
            JobStatusUpdate(
                job_id="job-12345",
                status=JobStatus.RUNNING,
                total_epochs=0,
                updated_time=datetime.now(UTC),
            )

    def test_default_values(self):
        """Test default values for job status update."""
        update = JobStatusUpdate(
            job_id="job-12345", status=JobStatus.PENDING, updated_time=datetime.now(UTC)
        )

        assert update.progress_percentage is None
        assert update.current_epoch is None
        assert update.total_epochs is None
        assert update.metrics == {}
        assert update.logs == []
        assert update.error_message is None
