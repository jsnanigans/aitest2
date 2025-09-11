# Weight Stream Processor - Agent Guidelines

## Build & Test Commands
```bash
uv pip install -r requirements.txt              # Install dependencies
uv run python main.py [csv_file]                # Run processor
uv run python tests/test_filename.py            # Run single test
uv run python -m pytest tests/                  # Run all tests (if pytest installed)
pyright                                         # Type check (mode: off)
```

## Code Style
- **Python 3.11+**, type hints optional (pyright mode: off)
- **Imports**: stdlib → third-party (numpy, pykalman, matplotlib) → local
- **NO COMMENTS** unless critical for complex math
- **Naming**: snake_case functions/vars, PascalCase classes
- **Line length**: 100 chars max
- **Test files**: Always in `tests/` with `test_` prefix

## Architecture (CRITICAL)
- **processor.py**: STATELESS, all @staticmethod methods
- **processor_database.py**: State persistence (in-memory)
- **Never split core files**: main.py, processor.py, visualization.py
- **State pattern**: load → process → save for EVERY measurement
- **No instance variables**: WeightProcessor has NO state