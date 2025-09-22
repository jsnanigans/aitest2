# Makefile for local Lambda testing

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ========== Docker & LocalStack ==========

.PHONY: docker-up
docker-up: ## Start Docker services (LocalStack, DynamoDB)
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@echo "Services started. DynamoDB Admin: http://localhost:8001"

.PHONY: docker-down
docker-down: ## Stop Docker services
	docker-compose down

.PHONY: docker-logs
docker-logs: ## Show Docker logs
	docker-compose logs -f

.PHONY: docker-reset
docker-reset: docker-down ## Reset Docker services (clean state)
	docker-compose down -v
	rm -rf data/dynamodb/*
	docker-compose up -d

# ========== SAM Local ==========

.PHONY: sam-build
sam-build: ## Build SAM application
	sam build --use-container

.PHONY: sam-local-api
sam-local-api: ## Start SAM Local API Gateway
	sam local start-api --config-file samconfig.toml

.PHONY: sam-local-lambda
sam-local-lambda: ## Start SAM Local Lambda runtime
	sam local start-lambda --config-file samconfig.toml

.PHONY: sam-invoke
sam-invoke: ## Invoke Lambda function with test event
	sam local invoke WeightProcessorFunction -e tests/local/events/process-single.json

.PHONY: sam-debug
sam-debug: ## Start SAM Local API with debugging
	sam local start-api --debug-port 5678 --config-file samconfig.toml

# ========== Direct Python Testing ==========

.PHONY: test-local
test-local: ## Run local Lambda tests with Python
	python local-test.py

.PHONY: test-local-interactive
test-local-interactive: ## Run interactive Lambda tester
	python local-test.py --interactive

.PHONY: test-local-mock
test-local-mock: ## Run tests with mocked DynamoDB
	python local-test.py --dynamodb mock

.PHONY: test-local-docker
test-local-docker: docker-up ## Run tests against Docker DynamoDB
	python local-test.py --dynamodb local

.PHONY: test-event
test-event: ## Test with specific event file
	@read -p "Enter event file path: " event_file; \
	python local-test.py --event $$event_file

# ========== Integration Testing ==========

.PHONY: test-integration
test-integration: docker-up ## Run integration tests
	pytest tests/integration/test_api_local.py -v

.PHONY: test-all
test-all: ## Run all tests (unit + integration)
	pytest tests/ -v --cov=src --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest tests/unit/ -v

# ========== Event Generation ==========

.PHONY: generate-events
generate-events: ## Generate test event files
	@mkdir -p test-events
	python -c "from tests.local.mock_events import save_events_to_files; save_events_to_files()"
	@echo "Test events saved to test-events/"

.PHONY: show-event
show-event: ## Show example event structure
	@python -c "from tests.local.mock_events import get_process_event_single; import json; print(json.dumps(get_process_event_single(), indent=2, default=str))"

# ========== Database Management ==========

.PHONY: db-scan
db-scan: ## Scan DynamoDB table
	aws dynamodb scan \
		--table-name weight-processor-state \
		--endpoint-url http://localhost:8000 \
		--region us-east-1

.PHONY: db-seed
db-seed: ## Seed test data to DynamoDB
	python scripts/seed-local-db.py

.PHONY: db-clear
db-clear: ## Clear DynamoDB table
	aws dynamodb scan \
		--table-name weight-processor-state \
		--endpoint-url http://localhost:8000 \
		--region us-east-1 \
		--query 'Items[*].[userId.S, stateType.S]' \
		--output text | while read userId stateType; do \
		aws dynamodb delete-item \
			--table-name weight-processor-state \
			--key "{\"userId\":{\"S\":\"$$userId\"},\"stateType\":{\"S\":\"$$stateType\"}}" \
			--endpoint-url http://localhost:8000 \
			--region us-east-1; \
	done

# ========== API Testing ==========

.PHONY: api-test-process
api-test-process: ## Test process endpoint via curl
	@curl -X POST http://localhost:3000/api/v1/process/test-user \
		-H "Content-Type: application/json" \
		-d @test-events/process_single.json | jq

.PHONY: api-test-cleanup
api-test-cleanup: ## Test cleanup endpoint via curl
	@curl -X POST http://localhost:3000/api/v1/cleanup/test-user \
		-H "Content-Type: application/json" \
		-d @test-events/cleanup.json | jq

.PHONY: api-test-state
api-test-state: ## Test get state endpoint via curl
	@curl -X GET http://localhost:3000/api/v1/state/test-user | jq

# ========== Development Utilities ==========

.PHONY: install
install: ## Install dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

.PHONY: format
format: ## Format code with black
	black src/ tests/

.PHONY: lint
lint: ## Lint code with flake8
	flake8 src/ tests/

.PHONY: type-check
type-check: ## Type check with mypy
	mypy src/

.PHONY: clean
clean: ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf test-events/
	rm -rf logs/

# ========== Quick Start ==========

.PHONY: setup
setup: install generate-events ## Initial setup

.PHONY: run
run: docker-up sam-local-api ## Start everything for local development

.PHONY: stop
stop: docker-down ## Stop everything

# ========== Performance Testing ==========

.PHONY: load-test
load-test: ## Run load test against local API
	python scripts/load-test.py --endpoint http://localhost:3000 --users 10 --duration 60

.PHONY: profile
profile: ## Profile Lambda function
	python -m cProfile -o profile.stats local-test.py
	python -m pstats profile.stats

# ========== Debugging ==========

.PHONY: debug-handler
debug-handler: ## Debug Lambda handler interactively
	python -m pdb -c "from src.lambda_handler import handler; import json; from tests.local.mock_events import get_process_event_single; handler(get_process_event_single(), None)"

.PHONY: debug-service
debug-service: ## Debug service layer interactively
	python -i -c "from src.services.weight_processor_service import WeightProcessorService; service = WeightProcessorService(); print('Service loaded as: service')"

# ========== Documentation ==========

.PHONY: docs
docs: ## Generate documentation
	@echo "API Documentation:"
	@echo "  POST   /api/v1/process/{userId}  - Process measurements"
	@echo "  POST   /api/v1/cleanup/{userId}  - Cleanup historical data"
	@echo "  POST   /api/v1/replay/{userId}   - Replay measurements"
	@echo "  GET    /api/v1/state/{userId}    - Get user state"
	@echo "  DELETE /api/v1/state/{userId}    - Delete user state"

# Default target
.DEFAULT_GOAL := help