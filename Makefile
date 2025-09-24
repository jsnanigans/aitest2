# Run main processing with test sample data (starts DynamoDB Local automatically)
run: db-start
	@echo "Waiting for DynamoDB Local to be ready..."
	@for i in {1..10}; do \
		if curl -s http://localhost:8000 > /dev/null 2>&1; then \
			echo "✓ DynamoDB Local is ready"; \
			break; \
		fi; \
		if [ $$i -eq 10 ]; then \
			echo "Error: DynamoDB Local failed to start"; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py && \
	uv run python main.py

# Run with a specific data file (starts DynamoDB Local automatically)
run-file: db-start
	@echo "Waiting for DynamoDB Local to be ready..."
	@for i in {1..10}; do \
		if curl -s http://localhost:8000 > /dev/null 2>&1; then \
			echo "✓ DynamoDB Local is ready"; \
			break; \
		fi; \
		if [ $$i -eq 10 ]; then \
			echo "Error: DynamoDB Local failed to start"; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py && \
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

# Install boto3 if not already installed
install-boto3:
	@uv run python -c "import boto3" 2>/dev/null || uv pip install boto3

# Build Lambda package locally (default - no containers)
local-build:
	sam build --template template-local.yaml

# Build Lambda package for production (with auth)
# build:
# 	sam build --template template.yaml

# Build for production deployment (Lambda only, no API Gateway)
# build-prod:
# 	@echo "🔨 Building Lambda package for production (Lambda only)..."
# 	sam build --template template-prod.yaml

# Deploy to AWS (dev) - includes API Gateway for testing
# deploy-dev:
# 	@echo "📦 Deploying to dev with API Gateway..."
# 	sam deploy --guided --template template.yaml --stack-name weight-processor-dev --parameter-overrides Environment=dev

# Deploy to production - Lambda only, no API Gateway
# deploy-prod: build-prod
# 	@echo "🚀 Deploying to production (Lambda only, no API Gateway)..."
# 	sam deploy --guided --template template-prod.yaml --stack-name weight-processor-prod --parameter-overrides Environment=prod

# Start local API Gateway (no Docker)
local-run:
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam local start-api --port 5448 --template .aws-sam/build/template.yaml

# View Lambda logs (local)
local-logs:
	@tail -50 .aws-sam/local/logs/*.log 2>/dev/null || echo "No logs found. Start the API first with 'make local'"

# Clean up generated files
clean:
	rm -rf .aws-sam
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Start DynamoDB Local and Admin UI
db-start:
	@if ! docker ps | grep -q weight-processor-dynamodb; then \
		echo "Starting DynamoDB Local..."; \
		docker-compose up -d dynamodb-local dynamodb-admin; \
	else \
		echo "DynamoDB Local is already running"; \
	fi

# Stop DynamoDB Local
db-stop:
	docker-compose down

# View DynamoDB data in browser
db-admin:
	@echo "Opening DynamoDB Admin UI..."
	@open http://localhost:8001 || xdg-open http://localhost:8001 || echo "Please open http://localhost:8001 in your browser"

# Clear DynamoDB data (restart containers)
db-clear:
	docker-compose down
	docker-compose up -d dynamodb-local dynamodb-admin
	@echo "DynamoDB data cleared"
