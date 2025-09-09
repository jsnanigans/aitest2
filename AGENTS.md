# Agent Instructions for Weight Stream Processor

## Project Structure
```
.
├── main.py                 # Entry point - NEVER split this
├── processor.py            # Core processor - NEVER split this  
├── visualization.py        # Visualization - NEVER split this
├── config.toml            # Configuration file
├── tests/                 # ALL test files go here
│   └── test_*.py         # Test files prefixed with test_
├── data/                  # Input CSV files
└── output/                # Generated results and visualizations
```

## Build & Test Commands
```bash
# Install dependencies (using uv package manager)
uv pip install -r requirements.txt

# Run main processor
uv run python main.py [csv_file]

# Run tests (all test files in tests/ directory)
uv run python tests/test_progress.py
uv run python tests/test_viz_improvements.py

# Type checking (if needed)
pyright
```

## Code Style Guidelines
- **Python 3.11+**, type hints optional (pyright typeCheckingMode: off)
- **Imports**: Standard library first, then third-party (numpy, pykalman, matplotlib), then local
- **Line length**: 100 chars max (per pyproject.toml)
- **Docstrings**: Brief module docstrings, class/method descriptions for complex logic
- **Architecture**: Keep it simple - 3 main files (main.py, processor.py, visualization.py)
- **NO COMMENTS** unless absolutely critical for understanding complex math
- **Error handling**: Basic validation only (weight 30-400kg, physiological limits)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **State management**: Streaming approach - O(1) memory per user, no full dataset in memory
- **Test files**: ALWAYS place new test files in `tests/` directory with `test_` prefix
- **Never split core files**: The 3 main files should remain single files