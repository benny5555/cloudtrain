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


# Test fixtures
@pytest.fixture
def cli_runner():
    """Provide CLI runner for tests."""
    return CliRunner()


@pytest.fixture
def sample_job_spec():
    """Provide sample job specification for tests."""
    return {
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


# Test functions for CLI commands (one-to-one mapping with source functions)


def test_cli_help(cli_runner):
    """Test cli() function help output."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CloudTrain - Universal Cloud Training API CLI" in result.output


def test_cli_verbose_flag(cli_runner):
    """Test cli() function verbose flag."""
    with patch("cloudtrain.cli.ConfigManager") as mock_config:
        mock_config.return_value = Mock()
        result = cli_runner.invoke(cli, ["--verbose", "providers"])
        assert "Verbose mode enabled" in result.output


def test_providers(cli_runner):
    """Test providers() command function."""
    with patch("cloudtrain.cli.CloudTrainingAPI") as mock_api:
        # Create a mock that can be used as an async context manager
        mock_api_instance = Mock()
        mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
        mock_api_instance.__aexit__ = AsyncMock(return_value=None)
        # get_available_providers is synchronous, not async
        mock_api_instance.get_available_providers.return_value = [CloudProvider.MOCK]
        mock_api.return_value = mock_api_instance

        with patch("cloudtrain.cli.ConfigManager") as mock_config:
            mock_config_manager = Mock()
            mock_config_manager.get_provider_config.return_value = Mock(
                region="us-east-1"
            )
            mock_config.return_value = mock_config_manager
            result = cli_runner.invoke(cli, ["providers"])
            assert result.exit_code == 0
            assert "Available Cloud Providers" in result.output


def test_submit_success(cli_runner, sample_job_spec):
    """Test submit() command function - successful job submission."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_job_spec, f)
        f.flush()  # Ensure data is written to disk
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
                mock_api_instance.get_available_providers.return_value = [
                    CloudProvider.MOCK
                ]
                mock_api_instance.submit_job = AsyncMock(return_value=mock_result)
                mock_api.return_value = mock_api_instance

                with patch("cloudtrain.cli.ConfigManager") as mock_config:
                    mock_config.return_value = Mock()
                    result = cli_runner.invoke(
                        cli, ["submit", job_spec_file, "--provider", "mock"]
                    )
                    assert result.exit_code == 0
                    assert "Job Submitted Successfully" in result.output
                    assert "test-job-123" in result.output
        finally:
            Path(job_spec_file).unlink()


def test_submit_invalid_file(cli_runner):
    """Test submit() command function - invalid file."""
    result = cli_runner.invoke(
        cli, ["submit", "nonexistent.json", "--provider", "mock"]
    )
    assert result.exit_code != 0


def test_submit_invalid_json(cli_runner):
    """Test submit() command function - invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json")
        invalid_file = f.name

    try:
        with patch("cloudtrain.cli.ConfigManager") as mock_config:
            mock_config.return_value = Mock()
            result = cli_runner.invoke(
                cli, ["submit", invalid_file, "--provider", "mock"]
            )
            assert result.exit_code == 1
            assert "Error loading job specification" in result.output
    finally:
        Path(invalid_file).unlink()


def test_status_success(cli_runner):
    """Test status() command function - successful status retrieval."""
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
            result = cli_runner.invoke(
                cli, ["status", "test-job-123", "--provider", "mock"]
            )
            assert result.exit_code == 0
            assert "test-job-123" in result.output


def test_list_jobs_success(cli_runner):
    """Test list_jobs() command function - successful job listing."""
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
            result = cli_runner.invoke(cli, ["list-jobs", "--provider", "mock"])
            assert result.exit_code == 0
            assert "Jobs" in result.output


def test_display_job_status():
    """Test display_job_status() utility function."""
    from cloudtrain.cli import display_job_status

    mock_status = Mock()
    mock_status.job_id = "test-job-123"
    mock_status.status = JobStatus.RUNNING
    mock_status.progress_percentage = 50.0
    mock_status.updated_time = "2023-01-01T00:00:00Z"
    mock_status.current_epoch = 5
    mock_status.total_epochs = 10
    mock_status.metrics = {"loss": 0.5, "accuracy": 0.85}
    mock_status.error_message = None
    mock_status.logs = ["Training started", "Epoch 1 completed"]

    # This function prints to console, so we just test it doesn't raise
    display_job_status(mock_status)


def test_main():
    """Test main() entry point function."""
    from cloudtrain.cli import main

    with patch("cloudtrain.cli.cli") as mock_cli:
        main()
        mock_cli.assert_called_once()
