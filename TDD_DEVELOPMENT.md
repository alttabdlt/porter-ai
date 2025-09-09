# 🧪 Test-Driven Development (TDD) for Porter.AI

## Overview

Porter.AI follows **Test-Driven Development (TDD)** principles to ensure code quality, maintainability, and cross-platform compatibility.

## 🏗️ Test Architecture

```
tests/
├── __init__.py
├── vlm_processors/           # VLM processor tests
│   ├── __init__.py
│   └── test_omnivlm_processor.py
├── capture/                  # Screen capture tests
│   ├── __init__.py
│   └── test_cross_platform_capture.py
├── integration/              # Integration tests
│   ├── __init__.py
│   └── test_main_pipeline.py
└── utils/                    # Test utilities
    ├── __init__.py
    └── mock_data.py          # Mock data generators
```

## 📝 TDD Workflow

### 1. Write Tests First
```python
# Example: Test VLM processor initialization
def test_processor_initialization(self, config):
    processor = OmniVLMProcessor(config)
    assert processor.config == config
    assert not processor.initialized
```

### 2. Run Tests (Expect Failure)
```bash
pytest tests/vlm_processors/test_omnivlm_processor.py -v
# ❌ Tests fail (expected - no implementation yet)
```

### 3. Implement Minimum Code
```python
class OmniVLMProcessor(BaseVLMProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.initialized = False
```

### 4. Run Tests Again
```bash
pytest tests/vlm_processors/test_omnivlm_processor.py -v
# ✅ Tests pass
```

### 5. Refactor & Repeat

## 🧪 Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Location**: `tests/vlm_processors/`, `tests/capture/`
- **Run**: `pytest -m unit`

### Integration Tests
- **Purpose**: Test component interactions
- **Location**: `tests/integration/`
- **Run**: `pytest -m integration`

### Performance Benchmarks
- **Purpose**: Measure performance metrics
- **Markers**: `@pytest.mark.benchmark`
- **Run**: `pytest -m benchmark`

## 🎯 Test Coverage

### Current Test Coverage
- **VLM Processors**: 60+ tests covering all methods
- **Screen Capture**: 40+ tests for cross-platform capture
- **Integration**: 25+ tests for pipeline flow
- **Mock Data**: Comprehensive mock generators

### Coverage Goals
- Minimum: 70% coverage (enforced in CI)
- Target: 85% coverage for critical paths
- 100% coverage for public APIs

## 🛠️ Running Tests

### Quick Start
```bash
# Install dependencies
pip install -r requirements-base.txt
pip install pytest pytest-cov pytest-asyncio pytest-benchmark

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific suite
python run_tests.py --suite vlm --coverage
```

### Test Runner Options
```bash
# Run all tests
python run_tests.py --suite all

# Run unit tests only
python run_tests.py --suite unit

# Run with coverage report
python run_tests.py --coverage

# Watch mode (auto-rerun on changes)
python run_tests.py --watch

# Stop on first failure
python run_tests.py --failfast
```

## 🎭 Mock Data Generators

### Screen Types
```python
from tests.utils.mock_data import MockScreenGenerator, ScreenType

generator = MockScreenGenerator()

# Generate different screen types
code_editor = generator.generate_screen(ScreenType.CODE_EDITOR)
terminal = generator.generate_screen(ScreenType.TERMINAL)
browser = generator.generate_screen(ScreenType.BROWSER)
```

### Mock VLM Outputs
```python
from tests.utils.mock_data import MockVLMOutput

description = MockVLMOutput.generate_description(ScreenType.CODE_EDITOR)
# "User is editing Python code in Visual Studio Code with syntax highlighting"

risk = MockVLMOutput.generate_risk_assessment()
# {"risk_level": "low", "reason": "No sensitive information detected"}
```

## 📊 Test Markers

```python
# Mark test categories
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.benchmark     # Performance benchmark
@pytest.mark.slow          # Slow test
@pytest.mark.requires_display  # Needs display
@pytest.mark.requires_gpu  # Needs GPU
```

Run specific markers:
```bash
pytest -m "unit and not slow"
pytest -m "integration and not requires_gpu"
```

## 🔄 Continuous Testing

### GitHub Actions (Future)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - uses: actions/checkout@v2
      - run: pip install -r requirements-base.txt
      - run: pytest --cov=app
```

### Pre-commit Hooks (Future)
```yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## 🐛 Debugging Tests

### Verbose Output
```bash
pytest -vv tests/capture/test_cross_platform_capture.py
```

### Debug with pdb
```python
def test_something():
    import pdb; pdb.set_trace()
    # Debug here
```

### Show print statements
```bash
pytest -s tests/
```

## 📈 Test Metrics

### Performance Benchmarks
```bash
pytest -m benchmark --benchmark-only
```

Output:
```
test_single_frame_capture_speed: 0.0234s (mean)
test_batch_processing_speed: 0.1562s (mean)
test_frame_encoding_speed: 0.0089s (mean)
```

### Coverage Report
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## ✅ Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mocks for APIs, models, hardware
3. **Clear Test Names**: `test_<what>_<condition>_<expected>`
4. **Arrange-Act-Assert**: Structure tests clearly
5. **Fast Tests**: Keep unit tests under 100ms
6. **Comprehensive Fixtures**: Reuse test setup
7. **Async Testing**: Use `pytest-asyncio` for async code

## 🚀 Benefits of TDD

1. **Confidence**: Changes don't break existing functionality
2. **Documentation**: Tests document expected behavior
3. **Design**: Forces better API design
4. **Refactoring**: Safe to refactor with test coverage
5. **Cross-platform**: Ensures compatibility across OS
6. **Performance**: Benchmarks catch regressions

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [TDD Best Practices](https://testdriven.io/blog/modern-tdd/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Async Testing Guide](https://pytest-asyncio.readthedocs.io/)

---

*Following TDD ensures Porter.AI remains robust, maintainable, and cross-platform compatible!*