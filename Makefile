.PHONY: help install test test-v test-replay test-lambda build build-local local local-health local-test deploy deploy-staging deploy-prod clean run run-file benchmark format lint create-filtered create-filtered-all report
.PHONY: docker-build docker-run docker-test docker-logs docker-invoke docker-stop docker-clean

# Default - show available commands
help:
	@echo "🚀 AWS Lambda Local Development Commands:"
	@echo ""
	@echo "  === Local Testing (Default - No Docker) ==="
	@echo "  make build-local    Build Lambda locally (no containers)"
	@echo "  make local          Start local API Gateway (http://localhost:5448)"
	@echo "  make local-health   Check API health status"
	@echo "  make local-test     Test process endpoint with curl"
	@echo "  make local-logs     View Lambda logs"
	@echo "  make clean          Clean up generated files"
	@echo ""
	@echo "  === Docker Local Testing (Alternative) ==="
	@echo "  make docker-build   Build Lambda with Docker containers"
	@echo "  make docker-run     Start local API with Docker"
	@echo "  make docker-health  Check API health (Docker)"
	@echo "  make docker-test    Test process endpoint (Docker)"
	@echo "  make docker-logs    View container logs"
	@echo "  make docker-stop    Stop all SAM containers"
	@echo "  make docker-clean   Clean Docker resources"
	@echo ""
	@echo "  === AWS Deployment ==="
	@echo "  make deploy-dev     Deploy to dev (with API Gateway for testing)"
	@echo "  make deploy-prod    Deploy to production (Lambda only, no API Gateway)"
	@echo ""
	@echo "  === Local Development ==="
	@echo "  make run            Run main processing with test sample data"
	@echo "  make run-file FILE=<file>  Run with a specific data file"
	@echo "  make test           Run all tests"
	@echo "  make test-v         Run tests with verbose output"
	@echo "  make test-lambda    Run Lambda handler tests"
	@echo "  make lint           Lint code"
	@echo "  make format         Format code with ruff"

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

# Build Lambda package locally (default - no containers)
build-local:
	@echo "🔨 Building Lambda package locally (no containers)..."
	sam build --template template-local.yaml

# Build Lambda package for production (with auth)
build:
	sam build --template template.yaml

# Build for production deployment (Lambda only, no API Gateway)
build-prod:
	@echo "🔨 Building Lambda package for production (Lambda only)..."
	sam build --template template-prod.yaml

# Deploy to AWS (dev) - includes API Gateway for testing
deploy-dev:
	@echo "📦 Deploying to dev with API Gateway..."
	sam deploy --guided --template template.yaml --stack-name weight-processor-dev --parameter-overrides Environment=dev

# Deploy to production - Lambda only, no API Gateway
deploy-prod: build-prod
	@echo "🚀 Deploying to production (Lambda only, no API Gateway)..."
	sam deploy --guided --template template-prod.yaml --stack-name weight-processor-prod --parameter-overrides Environment=prod

# Start local API Gateway (no Docker)
local: build-local
	@echo "🚀 Starting local API Gateway at http://localhost:5448"
	@echo "📝 API Endpoints (No Auth Required Locally):"
	@echo "  GET  http://localhost:5448/api/v1/health              - Health check"
	@echo "  POST http://localhost:5448/api/v1/process/{userId}    - Process measurements"
	@echo "  POST http://localhost:5448/api/v1/replay/{userId}     - Replay measurements"
	@echo "  POST http://localhost:5448/api/v1/cleanup/{userId}    - Cleanup with reset"
	@echo "  GET  http://localhost:5448/api/v1/state/{userId}      - Get user state"
	@echo ""
	@echo "Press Ctrl+C to stop..."
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam local start-api --port 5448 --template .aws-sam/build/template.yaml

# Test health endpoint locally (no Docker)
local-health:
	@echo "🏥 Checking API health..."
	curl -s http://localhost:5448/api/v1/health | python3 -m json.tool

# Test the process endpoint locally with sample data
local-test:
	@echo "📋 Testing process endpoint..."
	@curl -s -X POST http://localhost:5448/api/v1/process/test-user \
		-H "Content-Type: application/json" \
		-d '{"measurements": [{"uuid": "550e8400-e29b-41d4-a716-446655440000", "userId": "test-user", "weight": 75.5, "unit": "kg", "timestamp": "2024-01-01T10:00:00Z", "effectiveDateTime": "2024-01-01T10:00:00Z", "source": "patient-device"}]}' \
		| python3 -m json.tool

# View Lambda logs (local)
local-logs:
	@echo "📜 Showing recent Lambda logs..."
	@tail -50 .aws-sam/local/logs/*.log 2>/dev/null || echo "No logs found. Start the API first with 'make local'"

# Clean up generated files
clean:
	rm -rf .aws-sam
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ========== Docker Local Development Commands ==========

# Build Lambda package using Docker (no local Python needed)
docker-build:
	@echo "🔨 Building Lambda package with Docker (using local template - no auth)..."
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam build --use-container --template template-local.yaml

# Start local API Gateway on port 5448
docker-run: docker-build
	@echo "🚀 Starting local API Gateway at http://localhost:5448"
	@echo "📝 API Endpoints (No Auth Required Locally):"
	@echo "  GET  http://localhost:5448/api/v1/health              - Health check"
	@echo "  POST http://localhost:5448/api/v1/process/{userId}    - Process measurements"
	@echo "  POST http://localhost:5448/api/v1/replay/{userId}     - Replay measurements"
	@echo "  POST http://localhost:5448/api/v1/cleanup/{userId}    - Cleanup with reset"
	@echo "  GET  http://localhost:5448/api/v1/state/{userId}      - Get user state"
	@echo ""
	@echo "Press Ctrl+C to stop..."
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam local start-api --port 5448 --docker-network bridge --template template-local.yaml \
	--skip-pull-image --warm-containers EAGER

# Test health endpoint (no auth required locally)
docker-health:
	@echo "🏥 Checking API health..."
	curl -s http://localhost:5448/api/v1/health | jq .

# Test the process endpoint with sample data
docker-test:
	@echo "📋 Testing process endpoint..."
	curl -X POST http://localhost:5448/api/v1/process/test-user \
		-H "Content-Type: application/json" \
		-d '{"measurements": [{"uuid": "550e8400-e29b-41d4-a716-446655440000", "weight": 75.5, "unit": "kg", "effectiveDateTime": "2024-01-01T10:00:00Z", "source": "patient-device"}]}' \
		| jq .

# Test the replay endpoint
docker-test-replay:
	@echo "🔄 Testing replay endpoint..."
	curl -X POST http://localhost:5448/api/v1/replay/test-user \
		-H "Content-Type: application/json" \
		-d '{"replay_from_timestamp": "2024-01-01T00:00:00Z", "measurements": [{"uuid": "550e8400-e29b-41d4-a716-446655440001", "weight": 76.0, "unit": "kg", "effectiveDateTime": "2024-01-01T12:00:00Z", "source": "patient-device"}]}' \
		| jq .

# Invoke Lambda function directly with test event
docker-invoke:
	@echo "⚡ Invoking Lambda with test event..."
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam local invoke WeightProcessorFunction \
		--event test_events/process_measurements.json \
		--docker-network bridge \
		--template template-local.yaml

# Invoke with specific test event
docker-invoke-file:
	@echo "⚡ Invoking Lambda with $(EVENT)..."
	AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local AWS_SESSION_TOKEN=local \
	sam local invoke WeightProcessorFunction \
		--event $(EVENT) \
		--docker-network bridge \
		--template template-local.yaml

# View Docker container logs
docker-logs:
	@echo "📜 Showing Lambda container logs..."
	docker logs $$(docker ps -q --filter "label=com.amazonaws.sagemaker.local.mode=true" | head -1) 2>&1 | tail -50

# Stop all SAM local containers
docker-stop:
	@echo "🛑 Stopping SAM containers..."
	docker ps -q --filter "label=com.amazonaws.sagemaker.local.mode=true" | xargs -r docker stop
	docker ps -q --filter "ancestor=amazon/aws-sam-cli-emulation-image-python3.12" | xargs -r docker stop

# Clean up Docker resources
docker-clean: docker-stop
	@echo "🧹 Cleaning up Docker resources..."
	docker ps -aq --filter "label=com.amazonaws.sagemaker.local.mode=true" | xargs -r docker rm
	docker ps -aq --filter "ancestor=amazon/aws-sam-cli-emulation-image-python3.12" | xargs -r docker rm
	rm -rf .aws-sam

# Generate test event from live API call
docker-generate-event:
	@echo "📝 Generating test event template..."
	sam local generate-event apigateway aws-proxy \
		--method POST \
		--path /api/v1/process/test-user \
		--body '{"measurements": [{"uuid": "550e8400-e29b-41d4-a716-446655440000", "weight": 75.5, "unit": "kg", "effectiveDateTime": "2024-01-01T10:00:00Z", "source": "patient-device"}]}' \
		> test_events/generated_process.json
	@echo "Generated test_events/generated_process.json"

# Watch logs in real-time while API is running
docker-tail-logs:
	@echo "📜 Tailing Lambda logs (start API first with 'make docker-run')..."
	sam logs --tail