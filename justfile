# Stream Process Anchor - Just Commands

# Default - show available commands
default:
    @just --list

# Run main processing with test sample data
run:
    uv run python main.py

# Run with a specific data file
run-file file:
    uv run python main.py {{file}}

# Run all tests
test:
    uv run python -m pytest tests/ -q

# Run tests with verbose output
test-v:
    uv run python -m pytest tests/ -xvs

# Run replay tests specifically
test-replay:
    uv run python -m pytest tests/test_replay*.py -q

# Run performance benchmark
benchmark:
    uv run python scripts/measure_performance.py

# Clean up generated files
clean:
    rm -rf __pycache__ .pytest_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Install/update dependencies
install:
    uv sync

# Format code with ruff
format:
    uv run ruff format .

# Lint code
lint:
    uv run ruff check .
