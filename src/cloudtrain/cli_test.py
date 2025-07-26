"""Unit tests for CloudTrain CLI."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from cloudtrain.cli import cli
from cloudtrain.enums import CloudProvider, JobStatus
from cloudtrain.schemas import TrainingJobResult


class TestCLI:
    """Test CloudTrain CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create a sample job spec file
        self.job_spec = {
            "job_name": "test-job",
            "description": "Test job",
            "resource_requirements": {
                "instance_type": "gpu_small",
                "instance_count": 1,
            },
            "data_configuration": {
                "input_data_paths": ["s3://bucket/data"],
                "output_path": "s3://bucket/output",
            },
            "environment_configuration": {"entry_point": "train.py"},
        }

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CloudTrain - Universal Cloud Training API CLI" in result.output

    def test_cli_verbose_flag(self):
        """Test CLI verbose flag."""
        with patch("cloudtrain.cli.ConfigManager") as mock_config:
            mock_config.return_value = Mock()
            result = self.runner.invoke(cli, ["--verbose", "providers"])
            assert "Verbose mode enabled" in result.output

    def test_providers_command(self):
        """Test providers command."""
        with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
            # Create a mock that can be used as an async context manager
            mock_api_instance = Mock()
            mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
            mock_api_instance.__aexit__ = AsyncMock(return_value=None)
            # get_available_providers is synchronous, not async
            mock_api_instance.get_available_providers.return_value = [
                CloudProvider.MOCK
            ]
            mock_api.return_value = mock_api_instance

            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config_manager = Mock()
                mock_config_manager.get_provider_config.return_value = Mock(
                    region="us-east-1"
                )
                mock_config.return_value = mock_config_manager
                result = self.runner.invoke(cli, ["providers"])
                assert result.exit_code == 0
                assert "Available Cloud Providers" in result.output

    def test_submit_command_success(self):
        """Test successful job submission."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.job_spec, f)
            job_spec_file = f.name

        try:
            mock_result = TrainingJobResult(
                job_id="test-job-123",
                job_name="test-job",
                provider=CloudProvider.MOCK,
                status=JobStatus.PENDING,
                submission_time="2023-01-01T00:00:00Z",
            )

            with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
                mock_api_instance = Mock()
                mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
                mock_api_instance.__aexit__ = AsyncMock(return_value=None)
                mock_api_instance.get_available_providers = AsyncMock(
                    return_value=[CloudProvider.MOCK]
                )
                mock_api_instance.submit_job = AsyncMock(return_value=mock_result)
                mock_api.return_value = mock_api_instance

                with patch("cloudtrain.cli.ConfigManager") as mock_config:
                    mock_config.return_value = Mock()
                    result = self.runner.invoke(
                        cli, ["submit", job_spec_file, "--provider", "mock"]
                    )
                    assert result.exit_code == 0
                    assert "Job Submitted Successfully" in result.output
                    assert "test-job-123" in result.output
        finally:
            Path(job_spec_file).unlink()

    def test_submit_command_invalid_file(self):
        """Test job submission with invalid file."""
        result = self.runner.invoke(
            cli, ["submit", "nonexistent.json", "--provider", "mock"]
        )
        assert result.exit_code != 0

    def test_submit_command_invalid_json(self):
        """Test job submission with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json")
            invalid_file = f.name

        try:
            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config.return_value = Mock()
                result = self.runner.invoke(
                    cli, ["submit", invalid_file, "--provider", "mock"]
                )
                assert result.exit_code == 1
                assert "Error loading job specification" in result.output
        finally:
            Path(invalid_file).unlink()

    def test_status_command_success(self):
        """Test successful status retrieval."""
        mock_status = Mock()
        mock_status.job_id = "test-job-123"
        mock_status.status = JobStatus.RUNNING
        mock_status.progress_percentage = 50.0
        mock_status.updated_time = "2023-01-01T00:00:00Z"
        mock_status.current_epoch = None
        mock_status.total_epochs = None
        mock_status.metrics = {
            "loss": 0.5,
            "accuracy": 0.85,
        }  # Proper dict for iteration
        mock_status.error_message = None
        mock_status.logs = [
            "Training started",
            "Epoch 1 completed",
        ]  # Proper list for slicing

        with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
            mock_api_instance = Mock()
            mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
            mock_api_instance.__aexit__ = AsyncMock(return_value=None)
            mock_api_instance.get_job_status = AsyncMock(return_value=mock_status)
            mock_api.return_value = mock_api_instance

            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config.return_value = Mock()
                result = self.runner.invoke(
                    cli, ["status", "test-job-123", "--provider", "mock"]
                )
                assert result.exit_code == 0
                assert "test-job-123" in result.output

    def test_list_command_success(self):
        """Test successful job listing."""
        mock_job1 = Mock()
        mock_job1.job_id = "job-1"
        mock_job1.status = JobStatus.RUNNING
        mock_job1.progress_percentage = 50.0
        mock_job1.updated_time = datetime.now(UTC)

        mock_job2 = Mock()
        mock_job2.job_id = "job-2"
        mock_job2.status = JobStatus.COMPLETED
        mock_job2.progress_percentage = 100.0
        mock_job2.updated_time = datetime.now(UTC)

        mock_jobs = [mock_job1, mock_job2]

        with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
            mock_api_instance = Mock()
            mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
            mock_api_instance.__aexit__ = AsyncMock(return_value=None)
            mock_api_instance.list_jobs = AsyncMock(return_value=mock_jobs)
            mock_api.return_value = mock_api_instance

            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config.return_value = Mock()
                result = self.runner.invoke(cli, ["list-jobs", "--provider", "mock"])
                assert result.exit_code == 0
                assert "Jobs" in result.output

    def test_cancel_command_success(self):
        """Test successful job cancellation."""
        with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
            mock_api_instance = Mock()
            mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
            mock_api_instance.__aexit__ = AsyncMock(return_value=None)
            mock_api_instance.cancel_job = AsyncMock(return_value=True)
            mock_api.return_value = mock_api_instance

            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config.return_value = Mock()
                result = self.runner.invoke(
                    cli, ["cancel", "test-job-123", "--provider", "mock", "--yes"]
                )
                assert result.exit_code == 0
                assert "cancelled successfully" in result.output

    def test_config_command(self):
        """Test config command."""
        mock_config_manager = Mock()
        mock_config_manager.get_enabled_providers.return_value = [CloudProvider.MOCK]
        mock_config_manager.config_sources = ["config.yaml", "environment"]
        mock_config_manager.validate_configuration.return_value = {
            "valid": True,
            "providers": {"mock": {"enabled": True, "valid": True, "errors": []}},
            "errors": [],  # Add the missing errors key
        }

        with patch("cloudtrain.cli.ConfigManager") as mock_config:
            mock_config.return_value = mock_config_manager
            result = self.runner.invoke(cli, ["config"])
            assert result.exit_code == 0
            assert "Configuration" in result.output

    def test_submit_command_api_error(self):
        """Test job submission with API error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.job_spec, f)
            job_spec_file = f.name

        try:
            with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
                mock_api_instance = Mock()
                mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
                mock_api_instance.__aexit__ = AsyncMock(return_value=None)
                mock_api_instance.get_available_providers = AsyncMock(
                    return_value=[CloudProvider.MOCK]
                )
                mock_api_instance.submit_job = AsyncMock(
                    side_effect=Exception("API Error")
                )
                mock_api.return_value = mock_api_instance

                with patch("cloudtrain.cli.ConfigManager") as mock_config:
                    mock_config.return_value = Mock()
                    result = self.runner.invoke(
                        cli, ["submit", job_spec_file, "--provider", "mock"]
                    )
                    assert result.exit_code == 1
                    assert "Error submitting job" in result.output
        finally:
            Path(job_spec_file).unlink()

    def test_status_command_api_error(self):
        """Test status command with API error."""
        with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
            mock_api_instance = AsyncMock()
            mock_api_instance.__aenter__.return_value = mock_api_instance
            mock_api_instance.__aexit__.return_value = None
            mock_api_instance.get_job_status.side_effect = Exception("API Error")
            mock_api.return_value = mock_api_instance

            with patch("cloudtrain.cli.ConfigManager") as mock_config:
                mock_config.return_value = Mock()
                result = self.runner.invoke(
                    cli, ["status", "test-job-123", "--provider", "mock"]
                )
                assert result.exit_code == 1
                assert "Error getting job status" in result.output
