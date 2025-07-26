# CloudTrain: Universal Cloud Training API

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: pytype](https://img.shields.io/badge/type%20checked-pytype-blue.svg)](https://github.com/google/pytype)

CloudTrain is a Python library that provides a unified interface for submitting machine learning training jobs to multiple cloud providers. It abstracts away the complexity of different cloud APIs and provides a consistent, type-safe interface for training jobs across AWS SageMaker, Azure ML, Google Cloud AI Platform, and more.

## 🚀 Key Features

- **Universal API**: Single interface for multiple cloud providers
- **Type Safety**: Comprehensive type hints and validation with Pydantic
- **Async Support**: Non-blocking operations with async/await patterns
- **Plugin Architecture**: Easy to extend with new cloud providers
- **Configuration Management**: Secure credential and settings management
- **Job Monitoring**: Real-time status tracking and progress monitoring
- **Retry Logic**: Built-in retry mechanisms with exponential backoff
- **Mock Provider**: Testing and development without cloud credentials

## 🏗️ Supported Providers

| Provider | Status | Features |
|----------|--------|----------|
| AWS SageMaker | ✅ Planned | Job submission, monitoring, cancellation |
| Azure ML | ✅ Planned | Job submission, monitoring, cancellation |
| Google Cloud AI | ✅ Planned | Job submission, monitoring, cancellation |
| Alibaba Cloud PAI | ✅ Planned | Job submission, monitoring, cancellation |
| Tencent Cloud TI | ✅ Planned | Job submission, monitoring, cancellation |
| Mock Provider | ✅ Available | Full simulation for testing |

## 📦 Installation

### Basic Installation

```bash
pip install cloudtrain
```

### With Provider Dependencies

```bash
# AWS SageMaker
pip install cloudtrain[aws]

# Azure ML
pip install cloudtrain[azure]

# Google Cloud AI
pip install cloudtrain[gcp]

# All providers
pip install cloudtrain[all]

# Development dependencies
pip install cloudtrain[dev]
```

### Using uv (Recommended)

```bash
uv add cloudtrain
# or with extras
uv add cloudtrain[aws,azure,gcp]
```

## ⚙️ Configuration

CloudTrain supports multiple configuration methods:

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Azure Configuration
export AZURE_SUBSCRIPTION_ID=your_subscription_id
export AZURE_CLIENT_ID=your_client_id
export AZURE_CLIENT_SECRET=your_client_secret
export AZURE_TENANT_ID=your_tenant_id

# GCP Configuration
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your_project_id
```

### Configuration File

Create a `config.yaml` file:

```yaml
# CloudTrain Configuration
log_level: info
default_provider: aws

# Provider configurations
aws:
  enabled: true
  region: us-west-2

azure:
  enabled: true
  region: eastus

gcp:
  enabled: true
  region: us-central1

mock:
  enabled: true
  simulate_failures: false
```

### Programmatic Configuration

```python
from cloudtrain import CloudTrainingAPI
from cloudtrain.config import ConfigManager

# Custom configuration
config_manager = ConfigManager(config_file="my-config.yaml")
api = CloudTrainingAPI(config_manager=config_manager)
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from cloudtrain import (
    CloudTrainingAPI,
    CloudProvider,
    TrainingJobSpec,
    ResourceRequirements,
    DataConfiguration,
    EnvironmentConfiguration,
    InstanceType
)

async def main():
    # Initialize the API
    api = CloudTrainingAPI()

    # Create a job specification
    job_spec = TrainingJobSpec(
        job_name="my-training-job",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.GPU_SMALL,
            instance_count=1,
            volume_size_gb=50
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["s3://my-bucket/training-data/"],
            output_path="s3://my-bucket/output/"
        ),
        environment_configuration=EnvironmentConfiguration(
            entry_point="train.py",
            framework="pytorch",
            framework_version="2.0.0",
            python_version="3.9"
        )
    )

    # Submit the job
    result = await api.submit_job(CloudProvider.AWS, job_spec)
    print(f"Job submitted: {result.job_id}")

    # Monitor job status
    status = await api.get_job_status(CloudProvider.AWS, result.job_id)
    print(f"Job status: {status.status}")

    # Clean up
    await api.close()

# Run the example
asyncio.run(main())
```

### Using Mock Provider for Testing

```python
import asyncio
from cloudtrain import CloudTrainingAPI, CloudProvider
from cloudtrain.schemas import *

async def test_with_mock():
    api = CloudTrainingAPI()

    # Mock provider is always available for testing
    job_spec = TrainingJobSpec(
        job_name="test-job",
        resource_requirements=ResourceRequirements(
            instance_type=InstanceType.CPU_SMALL
        ),
        data_configuration=DataConfiguration(
            input_data_paths=["file:///tmp/data"],
            output_path="file:///tmp/output"
        ),
        environment_configuration=EnvironmentConfiguration(
            entry_point="train.py"
        )
    )

    # Submit to mock provider
    result = await api.submit_job(CloudProvider.MOCK, job_spec)
    print(f"Mock job submitted: {result.job_id}")

    await api.close()

asyncio.run(test_with_mock())
```

## ⚙️ Configuration

CloudTrain supports multiple configuration methods:

### Environment Variables

```bash
# AWS Configuration
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Azure Configuration
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_RESOURCE_GROUP=your-resource-group
export AZURE_WORKSPACE_NAME=your-workspace
export AZURE_TENANT_ID=your-tenant-id
export AZURE_CLIENT_ID=your-client-id
export AZURE_CLIENT_SECRET=your-client-secret

# GCP Configuration
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Configuration File

Create a `cloudtrain.yaml` file:

```yaml
# CloudTrain Configuration
log_level: info
default_provider: aws
auto_discover_providers: true

# Provider configurations
aws:
  enabled: true
  region: us-west-2
  timeout: 300
  max_retries: 3

azure:
  enabled: true
  region: eastus
  subscription_id: "your-subscription-id"
  resource_group: "your-resource-group"
  workspace_name: "your-workspace"

gcp:
  enabled: true
  project_id: "your-project-id"
  region: "us-central1"

mock:
  enabled: true
  simulate_failures: false
  failure_rate: 0.1
```

### Programmatic Configuration

```python
from cloudtrain import CloudTrainingAPI, ConfigManager
from cloudtrain.config import CloudTrainSettings, AWSConfig

# Create custom configuration
settings = CloudTrainSettings(
    log_level="debug",
    aws=AWSConfig(
        region="us-west-2",
        timeout=600
    )
)

config_manager = ConfigManager(settings=settings)
api = CloudTrainingAPI(config_manager=config_manager)
```

## 📊 Job Monitoring

CloudTrain provides comprehensive job monitoring capabilities:

```python
async def monitor_job(api, provider, job_id):
    """Monitor a training job until completion."""

    while True:
        status = await api.get_job_status(provider, job_id)

        print(f"Status: {status.status.value}")
        if status.progress_percentage:
            print(f"Progress: {status.progress_percentage:.1f}%")

        if status.metrics:
            print(f"Metrics: {status.metrics}")

        # Check if job is complete
        if status.status.is_terminal():
            print(f"Job finished with status: {status.status.value}")
            if status.error_message:
                print(f"Error: {status.error_message}")
            break

        # Wait before next check
        await asyncio.sleep(30)

# Usage
await monitor_job(api, CloudProvider.AWS, job_id)
```

## 🔧 Advanced Features

### Custom Instance Types

```python
# Use provider-specific instance types
resource_req = ResourceRequirements(
    instance_type=InstanceType.CUSTOM,
    custom_instance_type="ml.p4d.24xlarge",  # AWS-specific
    instance_count=2
)
```

### Provider-Specific Configuration

```python
job_spec = TrainingJobSpec(
    job_name="advanced-job",
    # ... other configuration ...
    provider_specific_config={
        # AWS SageMaker specific settings
        "RoleArn": "arn:aws:iam::123456789012:role/SageMakerRole",
        "VpcConfig": {
            "SecurityGroupIds": ["sg-12345678"],
            "Subnets": ["subnet-12345678"]
        }
    }
)
```

### Batch Job Management

```python
async def submit_multiple_jobs(api, job_specs):
    """Submit multiple jobs concurrently."""

    tasks = []
    for provider, spec in job_specs:
        task = api.submit_job(provider, spec)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Job {i} failed: {result}")
        else:
            print(f"Job {i} submitted: {result.job_id}")

    return results
```

## 🧪 Testing

CloudTrain follows a co-located testing approach with comprehensive test coverage:

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run tests without coverage
uv run pytest --no-cov

# Run only unit tests
uv run pytest -m unit

# Run integration tests (requires cloud credentials)
uv run pytest -m integration

# Run tests excluding slow tests
uv run pytest -m "not slow"

# Run specific test file
uv run pytest src/cloudtrain/api_test.py

# Run with verbose output
uv run pytest -v
```

### Test Structure

Tests are co-located with source code using the `*_test.py` naming convention:

```
src/cloudtrain/
├── api.py
├── api_test.py          # Tests for api.py
├── schemas.py
├── schemas_test.py      # Tests for schemas.py
└── utils/
    ├── validation.py
    └── validation_test.py  # Tests for validation.py
```

### Mock Provider Testing

The mock provider enables testing without cloud credentials:

```python
import pytest
from cloudtrain import CloudTrainingAPI, CloudProvider

@pytest.mark.asyncio
async def test_job_submission():
    api = CloudTrainingAPI()

    # Mock provider is always available
    assert CloudProvider.MOCK in api.get_available_providers()

    # Test job submission
    result = await api.submit_job(CloudProvider.MOCK, job_spec)
    assert result.job_id.startswith("mock-job-")

    await api.close()
```

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast tests with mocked dependencies
- **Integration Tests** (`@pytest.mark.integration`): Tests with real external dependencies
- **Slow Tests** (`@pytest.mark.slow`): Long-running tests (usually integration tests)

## 📚 Documentation

- **[API Reference](docs/api/)**: Detailed API documentation
- **[Provider Guides](docs/providers/)**: Provider-specific documentation
- **[Examples](examples/)**: Complete usage examples
- **[Architecture](docs/architecture.md)**: System architecture and design

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 🛠️ Development

### Environment Setup

CloudTrain uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

#### Installation

```bash
# Clone the repository
git clone https://github.com/cloudtrain/cloudtrain.git
cd cloudtrain

# Install with development dependencies
uv sync --extra dev

# Verify installation
uv run python -c "import cloudtrain; print(cloudtrain.__version__)"
```

### Development Workflow

#### Code Quality Standards

CloudTrain enforces strict code quality using automated tools. **Always run these in order**:

```bash
# 1. Sort imports (always run first)
uv run isort src/

# 2. Format code (always run second)
uv run black src/

# 3. Type checking (always run last)
uv run pytype src/cloudtrain/
```

#### Testing Workflow

```bash
# Run all tests with coverage
uv run pytest

# Run tests during development (faster)
uv run pytest --no-cov -x

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m "not slow"    # Exclude slow tests
```

#### Pre-commit Setup (Recommended)

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Code Quality Requirements

- **Import Sorting**: Use `isort` with black-compatible profile
- **Code Formatting**: Use `black` with 88-character line length
- **Type Checking**: Use `pytype` for static analysis
- **Test Coverage**: Minimum 80% coverage required
- **Testing**: Co-located tests using `*_test.py` naming convention

### Contribution Guidelines

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch from `main`
3. **Code Quality**: Ensure all quality checks pass
4. **Tests**: Write tests for new functionality
5. **Documentation**: Update documentation as needed
6. **Pull Request**: Submit PR with clear description

### Development Scripts

```bash
# Run full quality check
uv run isort src/ && uv run black src/ && uv run pytype src/cloudtrain/

# Run tests with coverage report
uv run pytest --cov-report=html

# Install package in development mode
uv pip install -e .
```

## 📄 License

CloudTrain is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

CloudTrain is inspired by excellent projects like:
- [AWS SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [MLflow](https://github.com/mlflow/mlflow)

## 📞 Support

- **Documentation**: [https://cloudtrain.readthedocs.io](https://cloudtrain.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/cloudtrain/cloudtrain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cloudtrain/cloudtrain/discussions)

---

**CloudTrain** - Simplifying machine learning training across clouds ☁️🚂