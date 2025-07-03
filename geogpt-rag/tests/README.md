# GeoGPT-RAG Test Suite

This directory contains comprehensive tests for the GeoGPT-RAG API application.

## 🧪 Test Overview

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions with mocked dependencies  
- **API Tests**: Test all FastAPI endpoints and HTTP functionality
- **Performance Tests**: Basic performance and concurrency testing
- **Error Handling**: Test error conditions and edge cases

### Test Coverage

The test suite covers:
- ✅ All API endpoints (`/`, `/health`, `/upload`, `/query`, `/retrieve`, `/collection`, `/stats`)
- ✅ File upload functionality (PDF, text files)
- ✅ Knowledge base operations (add, query, retrieve, drop)
- ✅ Error handling and edge cases
- ✅ Input validation and parameter limits
- ✅ Service availability checks
- ✅ Basic performance characteristics

## 🚀 Running Tests

### Prerequisites

Make sure you have installed the test dependencies:

```bash
cd geogpt-rag
pip install -r app/requirements.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run tests with verbose output
pytest -v

# Run tests with detailed coverage report
pytest --cov=app --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only API endpoint tests
pytest -m api tests/test_api.py::TestRootEndpoints
pytest -m api tests/test_api.py::TestFileUpload
pytest -m api tests/test_api.py::TestQueryEndpoints

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance

# Run specific test class
pytest tests/test_api.py::TestQueryEndpoints

# Run specific test method
pytest tests/test_api.py::TestFileUpload::test_upload_pdf_success
```

### Test Output Options

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html
# Report available at htmlcov/index.html

# Run with different verbosity levels
pytest -v          # verbose
pytest -vv         # very verbose  
pytest -q          # quiet

# Show test durations
pytest --durations=10

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## 📋 Test Structure

### Key Test Files

- `test_api.py` - Main API endpoint tests
- `pytest.ini` - Test configuration
- `conftest.py` - Shared test fixtures (if needed)

### Test Classes

1. **TestRootEndpoints** - Basic API endpoints
2. **TestFileUpload** - Document upload functionality
3. **TestQueryEndpoints** - Query and retrieval operations
4. **TestCollectionManagement** - Vector database operations
5. **TestStatisticsEndpoint** - Statistics and monitoring
6. **TestErrorHandling** - Error conditions
7. **TestIntegration** - End-to-end workflows
8. **TestPerformance** - Performance characteristics

## 🛠️ Test Configuration

### Pytest Configuration (`pytest.ini`)

- **Coverage**: Minimum 80% coverage required
- **Timeouts**: 5-minute timeout per test
- **Async Support**: Automatic asyncio mode
- **Markers**: Organized by test type
- **Logging**: Enabled during test runs

### Environment Variables for Testing

```bash
# Optional: Override test configuration
export PYTHONPATH=/path/to/geogpt-rag
export TEST_ENV=testing
```

## 🔧 Mocking Strategy

Tests use extensive mocking to avoid dependencies on:
- External LLM services (OpenAI API)
- Vector databases (Milvus/Zilliz)
- GPU-dependent models (embeddings, reranking)
- File system operations
- Network requests

This ensures tests run quickly and reliably in any environment.

## 📊 Coverage Reports

After running tests with coverage:

```bash
# View terminal coverage summary
pytest --cov=app --cov-report=term-missing

# Generate and view HTML report
pytest --cov=app --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🐛 Debugging Tests

```bash
# Run with Python debugger on failures
pytest --pdb

# Run single test with full output
pytest -vvs tests/test_api.py::TestFileUpload::test_upload_pdf_success

# Run with debug logging
pytest --log-cli-level=DEBUG

# Run tests that failed last time
pytest --lf
```

## 🔄 Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast test run for CI
pytest --maxfail=5 --tb=short

# With coverage and XML output for CI tools
pytest --cov=app --cov-report=xml --junitxml=test-results.xml
```

## 📝 Adding New Tests

When adding new functionality:

1. Add tests to the appropriate test class
2. Use appropriate pytest markers
3. Mock external dependencies
4. Include both success and failure cases
5. Test input validation
6. Update this README if needed

### Test Template

```python
@patch('app.main.kb_instance')
def test_new_feature(self, mock_kb):
    """Test new feature functionality."""
    # Setup mocks
    mock_kb.new_method.return_value = expected_result
    
    # Make request
    response = client.post("/new-endpoint", json=test_data)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["field"] == expected_value
    
    # Verify mocks called correctly
    mock_kb.new_method.assert_called_once_with(expected_params)
```

## 🏆 Best Practices

- Write tests before implementing features (TDD)
- Keep tests isolated and independent
- Use descriptive test names
- Test both success and failure paths
- Mock external dependencies
- Maintain high test coverage (>80%)
- Run tests locally before committing 