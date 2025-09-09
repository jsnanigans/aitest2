# AGENTS.md - Coding Guidelines for AI Agents

## Build & Test Commands
```bash
# Install dependencies using uv package manager
uv pip install -r requirements.txt
uv pip install -e ".[dev]"  # Include dev dependencies

# Run tests
uv run pytest tests/                     # All tests
uv run pytest tests/test_kalman.py       # Single test file
uv run pytest tests/ -v --cov=src        # With coverage

# Linting & formatting
uv run black src/ tests/                 # Format code (100 char line limit)
uv run ruff src/ tests/                  # Lint code
uv run mypy src/                         # Type checking (optional typing)
```

## Code Style Guidelines
- **NO COMMENTS** unless absolutely critical for understanding complex algorithms
- **Imports**: Standard library → third-party → local (absolute imports from src/)
- **Line length**: 100 characters max (configured in pyproject.toml)
- **Type hints**: Optional but encouraged for public APIs and complex functions
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Let exceptions bubble up, minimal try/except blocks
- **Performance first**: Optimize for streaming/memory efficiency over readability
- **Testing**: Test files in tests/ named test_*.py, use pytest fixtures
- **Config**: Use config.toml for all configuration, loaded via src.core.config_loader