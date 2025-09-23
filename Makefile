.PHONY: help install test test-v test-replay test-lambda build deploy deploy-staging deploy-prod local clean run run-file benchmark format lint create-filtered create-filtered-all report

# Default - show available commands
help:
	@echo "Available commands:"
	@echo "  make run            Run main processing with test sample data"
	@echo "  make run-file FILE=<file>  Run with a specific data file"
	@echo "  make create-filtered       Create filtered output (nocon)"
	@echo "  make create-filtered-all   Create filtered output (all)"
	@echo "  make report         Run report generation"
	@echo "  make test           Run all tests"
	@echo "  make test-v         Run tests with verbose output"
	@echo "  make test-replay    Run replay tests specifically"
	@echo "  make test-lambda    Run Lambda handler tests"
	@echo "  make benchmark      Run performance benchmark"
	@echo "  make install        Install/update dependencies"
	@echo "  make format         Format code with ruff"
	@echo "  make lint           Lint code"
	@echo "  make build          Build Lambda package"
	@echo "  make deploy         Deploy to AWS (dev)"
	@echo "  make deploy-staging Deploy to staging"
	@echo "  make deploy-prod    Deploy to production"
	@echo "  make local          Run Lambda locally"
	@echo "  make clean          Clean up generated files"

# Run main processing with test sample data
run:
	uv run python main.py

# Run with a specific data file
run-file:
	uv run python main.py $(FILE)

# Create filtered output for nocon data
create-filtered:
	uv run python main.py data/2025-09-05_nocon.csv --max-users 0 --no-viz --filtered-output data/2025-09-05_nocon_filtered.csv

# Create filtered output for all data
create-filtered-all:
	uv run python main.py data/2025-09-05_all.csv --max-users 0 --no-viz --filtered-output data/2025-09-05_all_filtered.csv

# Run report generation
report:
	cd create-report && uv run python run.py --employer APPLE_EMPLOYER --visualize

# Run all tests
test:
	uv run python -m pytest tests/ -q

# Run tests with verbose output
test-v:
	uv run python -m pytest tests/ -xvs

# Run replay tests specifically
test-replay:
	uv run python -m pytest tests/test_replay*.py -q

# Run Lambda handler tests
test-lambda:
	uv run python -m pytest tests/test_lambda_handler.py -xvs

# Run performance benchmark
benchmark:
	uv run python scripts/measure_performance.py

# Install/update dependencies
install:
	uv sync
	uv pip install -r requirements.txt
	uv pip install -r requirements-lambda.txt

# Format code with ruff
format:
	uv run ruff format .

# Lint code
lint:
	uv run ruff check .

# Build Lambda package
build:
	sam build --use-container

# Deploy to AWS (dev)
deploy:
	./scripts/deploy/deploy.sh --env dev

# Deploy to staging
deploy-staging:
	./scripts/deploy/deploy.sh --env staging

# Deploy to production
deploy-prod:
	./scripts/deploy/deploy.sh --env prod --auto-confirm

# Run Lambda locally
local:
	./scripts/deploy/test_local.sh

# Clean up generated files
clean:
	rm -rf .aws-sam
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete