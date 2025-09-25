# Weight Processor Service - Makefile
# =====================================

# Variables
COMPOSE_FILE := docker-compose.yml
COMPOSE_SAM_FILE := docker-compose.sam.yml
DOCKER_COMPOSE := docker-compose -f $(COMPOSE_FILE)
DOCKER_COMPOSE_SAM := docker-compose -f $(COMPOSE_SAM_FILE)

.PHONY: help setup run run-file \
        docker-up docker-down docker-restart docker-status docker-clean \
        docker-shell docker-logs docker-build \
        db-start db-stop db-reset db-admin db-clear \
        sam-build sam-local sam-deploy sam-invoke sam-logs \
        clean

# Default target - show help
help:
	@echo "Weight Processor Service - Commands"
	@echo "===================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup            - Install dependencies with uv"
	@echo "  make docker-up        - Start all Docker services"
	@echo "  make sam-local        - Start SAM API locally (port 3000)"
	@echo ""
	@echo "Local Development:"
	@echo "  make run              - Run with sample data"
	@echo "  make run-file FILE=x  - Run with specific data file"
	@echo ""
	@echo "Docker Environment:"
	@echo "  make docker-up        - Start Docker services (LocalStack, DynamoDB)"
	@echo "  make docker-down      - Stop Docker services"
	@echo "  make docker-restart   - Restart Docker services"
	@echo "  make docker-status    - Show container status"
	@echo "  make docker-shell     - Open shell in SAM container"
	@echo "  make docker-logs      - Show container logs"
	@echo "  make docker-clean     - Remove all containers and volumes"
	@echo ""
	@echo "Database Management:"
	@echo "  make db-start         - Start DynamoDB Local"
	@echo "  make db-stop          - Stop DynamoDB Local"
	@echo "  make db-reset         - Reset database and tables"
	@echo "  make db-admin         - Open DynamoDB Admin UI (port 8001)"
	@echo ""
	@echo "SAM Operations:"
	@echo "  make sam-build        - Build Lambda package"
	@echo "  make sam-local        - Start local API (port 3000)"
	@echo "  make sam-deploy       - Deploy to AWS (interactive)"
	@echo "  make sam-logs         - View Lambda logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove build artifacts"

# =====================================
# Setup & Installation
# =====================================

setup:
	@echo "📦 Setting up Python environment..."
	@uv venv
	@uv pip install -r requirements.txt
	@uv pip install -r requirements-lambda.txt
	@uv pip install -r requirements-dev.txt
	@echo "✅ Setup complete. Activate with: source .venv/bin/activate"

# =====================================
# Local Development
# =====================================

# Run with sample data
run: db-start
	@echo "🚀 Running with sample data..."
	@sleep 2  # Wait for DynamoDB
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py && \
	uv run python src/local/main.py

# Run with specific file
run-file: db-start
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: FILE parameter required. Usage: make run-file FILE=data/sample.csv"; \
		exit 1; \
	fi
	@echo "🚀 Processing $(FILE)..."
	@sleep 2  # Wait for DynamoDB
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py && \
	uv run python src/local/main.py $(FILE)

# =====================================
# Docker Environment Management
# =====================================

# Start all Docker services
docker-up:
	@echo "🚀 Starting Docker services..."
	@$(DOCKER_COMPOSE) up -d
	@echo "⏳ Waiting for services..."
	@sleep 5
	@echo "✅ Docker services ready!"
	@echo "   - LocalStack: http://localhost:4566"
	@echo "   - DynamoDB: http://localhost:8000"
	@echo "   - DynamoDB Admin: http://localhost:8001"

# Stop Docker services
docker-down:
	@echo "🛑 Stopping Docker services..."
	@$(DOCKER_COMPOSE) down

# Restart Docker services
docker-restart: docker-down docker-up

# Show container status
docker-status:
	@echo "📊 Container Status:"
	@$(DOCKER_COMPOSE) ps

# Open shell in container (if using SAM compose)
docker-shell:
	@echo "🖥️ Opening Docker shell..."
	@docker run -it --rm \
		-v $(PWD):/var/task \
		-w /var/task \
		--network weight-processor-net \
		python:3.12 bash

# View logs
docker-logs:
	@$(DOCKER_COMPOSE) logs -f

# Build Docker images
docker-build:
	@echo "🔨 Building Docker images..."
	@$(DOCKER_COMPOSE) build --no-cache

# Clean everything
docker-clean:
	@echo "🧹 Cleaning Docker environment..."
	@$(DOCKER_COMPOSE) down -v
	@docker system prune -f
	@echo "✅ Docker environment cleaned"

# =====================================
# Database Management
# =====================================

# Start DynamoDB Local
db-start:
	@if ! docker ps | grep -q weight-processor-dynamodb; then \
		echo "🗄️ Starting DynamoDB Local..."; \
		$(DOCKER_COMPOSE) up -d dynamodb-local dynamodb-admin; \
		sleep 3; \
		echo "✅ DynamoDB ready on port 8000"; \
		echo "📊 Admin UI: http://localhost:8001"; \
	else \
		echo "✅ DynamoDB already running"; \
	fi

# Stop DynamoDB
db-stop:
	@echo "🛑 Stopping DynamoDB..."
	@$(DOCKER_COMPOSE) stop dynamodb-local dynamodb-admin

# Reset database
db-reset: db-stop db-start
	@echo "🔄 Resetting database..."
	@sleep 3
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py
	@echo "✅ Database reset complete"

# Open DynamoDB Admin
db-admin:
	@echo "📊 Opening DynamoDB Admin..."
	@open http://localhost:8001 || xdg-open http://localhost:8001 || echo "Open http://localhost:8001"

# Clear database data
db-clear:
	@echo "🧹 Clearing database..."
	@$(DOCKER_COMPOSE) down dynamodb-local dynamodb-admin
	@$(DOCKER_COMPOSE) up -d dynamodb-local dynamodb-admin
	@echo "✅ Database cleared"

# =====================================
# SAM (Serverless Application Model)
# =====================================

# Build Lambda package
sam-build:
	@echo "🔨 Building SAM application..."
	@cd aws && sam build --template template.yaml
	@echo "✅ Build complete"

# Build for local testing
sam-build-local:
	@echo "🔨 Building for local testing..."
	@cd aws && sam build --template template-local.yaml
	@echo "✅ Local build complete"

# Start local API
sam-local: db-start sam-build-local
	@echo "🚀 Starting SAM Local API..."
	@echo "📡 API available at http://localhost:3000"
	@cd aws && sam local start-api \
		--port 3000 \
		--template .aws-sam/build/template.yaml \
		--parameter-overrides \
			Environment=local \
			DynamoDBEndpoint=http://localhost:8000

# Deploy to AWS (interactive)
sam-deploy: sam-build
	@echo "🚀 Deploying to AWS..."
	@cd aws && sam deploy --guided

# Deploy with specific environment
sam-deploy-env: sam-build
	@if [ -z "$(ENV)" ]; then \
		echo "❌ Error: ENV parameter required. Usage: make sam-deploy-env ENV=dev"; \
		exit 1; \
	fi
	@echo "🚀 Deploying to $(ENV) environment..."
	@cd aws && sam deploy --config-env $(ENV)

# Invoke Lambda function locally
sam-invoke:
	@if [ -z "$(FUNC)" ]; then \
		echo "❌ Error: FUNC parameter required. Usage: make sam-invoke FUNC=WeightProcessorFunction"; \
		exit 1; \
	fi
	@echo "⚡ Invoking $(FUNC)..."
	@cd aws && sam local invoke $(FUNC) --event ../test_events/process_event.json

# View Lambda logs
sam-logs:
	@if [ -z "$(STACK)" ]; then \
		echo "📋 Viewing local SAM logs..."; \
		tail -50 aws/.aws-sam/local/logs/*.log 2>/dev/null || echo "No logs found"; \
	else \
		echo "📋 Viewing logs for stack $(STACK)..."; \
		sam logs -n WeightProcessorFunction --stack-name $(STACK) --tail; \
	fi

# Delete AWS stack
sam-delete:
	@if [ -z "$(STACK)" ]; then \
		echo "❌ Error: STACK parameter required. Usage: make sam-delete STACK=weight-processor-dev"; \
		exit 1; \
	fi
	@echo "🗑️ Deleting stack $(STACK)..."
	@sam delete --stack-name $(STACK)

# =====================================
# Testing API Endpoints
# =====================================

# Test local API health
test-api:
	@echo "🏥 Testing API health..."
	@curl -s http://localhost:3000/health | jq '.' || echo "API not running. Start with: make sam-local"

# Test process endpoint
test-process:
	@echo "📊 Testing process endpoint..."
	@curl -X POST http://localhost:3000/process \
		-H "Content-Type: application/json" \
		-d '{ \
			"device_id": "scale-001", \
			"measurements": [ \
				{"weight": 75.5, "timestamp": "2024-01-01T10:00:00Z", "source": "patient-device"} \
			] \
		}' | jq '.'

# =====================================
# Cleanup
# =====================================

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf aws/.aws-sam
	@rm -rf __pycache__ .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf htmlcov .coverage
	@echo "✅ Cleanup complete"

# =====================================
# Utility Targets
# =====================================

# Show environment info
info:
	@echo "Environment Information:"
	@echo "========================"
	@python --version
	@docker --version
	@docker-compose --version
	@sam --version 2>/dev/null || echo "SAM CLI not installed"
	@uv --version 2>/dev/null || echo "uv not installed"