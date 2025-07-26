# CloudTrain Testing Guide

This document provides comprehensive information about testing in the CloudTrain project, including test structure, conventions, and execution guidelines.

## Test Structure

CloudTrain follows a co-located testing approach where test files are placed in the same directories as the source code they test. This structure provides several benefits:

- **Easy Navigation**: Tests are immediately adjacent to the code they test
- **Clear Ownership**: Each module's tests are clearly associated with that module
- **Simplified Imports**: Test files can easily import the modules they test
- **Atomic Changes**: Code and test changes can be made together

### File Naming Conventions

- **Unit Tests**: `*_test.py` (e.g., `enums_test.py`, `api_test.py`)
- **Integration Tests**: `*_integration_test.py` (e.g., `provider_integration_test.py`)
- **Test Fixtures**: `conftest.py` (shared fixtures and configuration)

### Directory Structure

```
src/cloudtrain/
├── __init__.py
├── enums.py
├── enums_test.py                    # Unit tests for enums
├── schemas.py
├── schemas_test.py                  # Unit tests for schemas
├── api.py
├── api_test.py                      # Unit tests for main API
├── config/
│   ├── __init__.py
│   ├── manager.py
│   ├── manager_test.py              # Unit tests for config manager
│   ├── settings.py
│   └── settings_test.py             # Unit tests for settings
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── base_test.py                 # Unit tests for base provider
│   └── mock/
│       ├── __init__.py
│       ├── provider.py
│       ├── provider_test.py         # Unit tests for mock provider
│       └── provider_integration_test.py  # Integration tests
└── utils/
    ├── __init__.py
    ├── validation.py
    ├── validation_test.py           # Unit tests for validation
    ├── retry.py
    └── retry_test.py                # Unit tests for retry logic

tests/
└── conftest.py                      # Global test fixtures and configuration
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)

Unit tests focus on testing individual components in isolation:

- **Scope**: Single functions, methods, or classes
- **Dependencies**: Mocked or stubbed external dependencies
- **Speed**: Fast execution (< 1 second per test)
- **Coverage**: High code coverage with edge cases

**Example**:
```python
@pytest.mark.unit
class TestCloudProvider:
    def test_provider_values(self):
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.AZURE.value == "azure"
```

### Integration Tests (`@pytest.mark.integration`)

Integration tests verify component interactions:

- **Scope**: Multiple components working together
- **Dependencies**: Real implementations where possible
- **Speed**: Moderate execution (1-10 seconds per test)
- **Coverage**: End-to-end workflows and API interactions

**Example**:
```python
@pytest.mark.integration
class TestMockProviderIntegration:
    @pytest.mark.asyncio
    async def test_full_job_lifecycle(self, mock_config_manager, sample_job_spec):
        async with CloudTrainingAPI(config_manager=mock_config_manager) as api:
            result = await api.submit_job(CloudProvider.MOCK, sample_job_spec)
            # ... test complete workflow
```

### Slow Tests (`@pytest.mark.slow`)

Tests that take longer to execute:

- **Scope**: Complex integration scenarios, performance tests
- **Dependencies**: May involve real cloud provider APIs (in CI/CD)
- **Speed**: Slow execution (> 10 seconds per test)
- **Coverage**: End-to-end system validation

## Test Execution

### Using pytest directly

```bash
# Run all unit tests
pytest -m unit

# Run all integration tests
pytest -m integration

# Run tests for specific module
pytest src/cloudtrain/enums_test.py

# Run with coverage
pytest --cov=src/cloudtrain --cov-report=html

# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

### Using the test runner script

```bash
# Run unit tests
python scripts/run_tests.py --type unit

# Run integration tests with coverage
python scripts/run_tests.py --type integration --coverage

# Run all tests in parallel, skipping slow ones
python scripts/run_tests.py --type all --parallel 4 --fast

# Run tests for specific provider
python scripts/run_tests.py --provider mock --verbose
```

## Test Fixtures

CloudTrain provides comprehensive test fixtures in `tests/conftest.py`:

### Configuration Fixtures

- `mock_config_manager`: Mock configuration manager for unit tests
- `test_config_manager`: Real configuration manager with test settings

### Data Fixtures

- `sample_job_spec`: Complete job specification for testing
- `sample_job_result`: Job submission result
- `sample_job_status_update`: Job status update
- `gpu_job_spec`: GPU-specific job specification
- `custom_instance_job_spec`: Custom instance type specification

### Usage Example

```python
def test_job_submission(sample_job_spec, mock_config_manager):
    # Fixtures are automatically injected
    api = CloudTrainingAPI(config_manager=mock_config_manager)
    # ... test implementation
```

## Testing Best Practices

### 1. Test Naming

- Use descriptive test names that explain what is being tested
- Follow the pattern: `test_<action>_<condition>_<expected_result>`

```python
def test_submit_job_with_invalid_provider_raises_error(self):
    # Clear what this test does
```

### 2. Test Organization

- Group related tests in classes
- Use descriptive class names: `TestCloudProvider`, `TestJobSubmission`
- Order tests logically (happy path first, then edge cases)

### 3. Assertions

- Use specific assertions with clear error messages
- Test both positive and negative cases
- Verify all relevant aspects of the result

```python
def test_job_result_structure(self, sample_job_result):
    assert sample_job_result.job_id.startswith("test-job-")
    assert sample_job_result.provider == CloudProvider.MOCK
    assert sample_job_result.status == JobStatus.PENDING
    assert sample_job_result.submission_time is not None
```

### 4. Mocking

- Mock external dependencies (cloud APIs, file system, network)
- Use `unittest.mock` for Python standard library
- Mock at the boundary of your system

```python
@patch('cloudtrain.api.retry_with_backoff')
async def test_submit_job_with_retry(self, mock_retry, sample_job_spec):
    mock_retry.return_value = sample_job_result
    # ... test implementation
```

### 5. Async Testing

- Use `@pytest.mark.asyncio` for async test functions
- Properly await async operations
- Use `AsyncMock` for mocking async functions

```python
@pytest.mark.asyncio
async def test_async_job_submission(self):
    mock_provider = AsyncMock()
    mock_provider.submit_job.return_value = expected_result
    # ... test implementation
```

## Coverage Requirements

CloudTrain maintains high test coverage standards:

- **Minimum Coverage**: 80% overall
- **Critical Modules**: 90%+ coverage (API, providers, validation)
- **New Code**: 100% coverage for new features

### Generating Coverage Reports

```bash
# HTML report
pytest --cov=src/cloudtrain --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=src/cloudtrain --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=src/cloudtrain --cov-fail-under=80
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled nightly runs

### Test Matrix

- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu, macOS, Windows
- **Test Types**: Unit (all platforms), Integration (Ubuntu only)

### Provider Testing

- **Mock Provider**: Always tested (no credentials required)
- **Cloud Providers**: Tested in CI with service account credentials
- **Credential Management**: Stored as GitHub Secrets

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test
pytest src/cloudtrain/api_test.py::TestCloudTrainingAPI::test_submit_job_success -v

# Run with debugger
pytest --pdb src/cloudtrain/api_test.py::TestCloudTrainingAPI::test_submit_job_success

# Run with print statements
pytest -s src/cloudtrain/api_test.py
```

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Async Issues**: Use `@pytest.mark.asyncio` and proper await syntax
3. **Mock Issues**: Verify mock paths and return values
4. **Fixture Issues**: Check fixture scope and dependencies

### Test Data

- Use fixtures for consistent test data
- Avoid hardcoded values in tests
- Create realistic but minimal test data

## Performance Testing

### Load Testing

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_job_submissions(self):
    # Test multiple concurrent operations
    tasks = [submit_job(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    # Verify all succeeded
```

### Memory Testing

```python
def test_memory_usage_within_limits(self):
    # Monitor memory usage during operations
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    # ... perform operations
    final_memory = process.memory_info().rss
    assert final_memory - initial_memory < MAX_MEMORY_INCREASE
```

## Contributing Tests

When contributing to CloudTrain:

1. **Write Tests First**: Follow TDD when possible
2. **Test Coverage**: Ensure new code has appropriate test coverage
3. **Test Types**: Include both unit and integration tests
4. **Documentation**: Update this guide for new testing patterns
5. **Review**: Have tests reviewed along with code changes

### Test Review Checklist

- [ ] Tests cover happy path and edge cases
- [ ] Test names are descriptive and clear
- [ ] Appropriate use of fixtures and mocks
- [ ] Async tests properly marked and implemented
- [ ] Integration tests use realistic scenarios
- [ ] Performance implications considered
- [ ] Documentation updated if needed
