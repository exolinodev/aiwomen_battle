# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install production: `pip install -e .`
- Install development tools: `pip install -e ".[dev]"`
- Run tests: `python -m pytest tests/`
- Run tests with coverage: `python -m pytest --cov=src tests/`
- Run single test: `python -m pytest tests/test_specific.py`
- Check types: `mypy src/`
- Run specific test case: `python -m pytest tests/test_file.py::TestClass::test_function`
- Lint code: `ruff check src/`
- Format code: `black src/ && ruff format src/`

## Code Style
- PEP 8 compliant Python code
- Type hints for all functions and classes
- Imports order: stdlib → third-party → project → relative
- Classes: PascalCase (VideoEditor)
- Methods/functions/variables: snake_case (process_video)
- Constants: UPPER_CASE
- Documentation: Google-style docstrings
- Error handling: Use custom exceptions hierarchy (WATWError base)

## Project Structure
- Core functionality: src/watw/core/
- Video processing: src/watw/core/video/
- API clients: src/watw/api/
- Utilities: src/watw/utils/
- Common utils: src/watw/utils/common/
- Tests in separate tests/ directory
- Output files stored in outputs/workflow_output/