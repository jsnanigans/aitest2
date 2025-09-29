# Weight Processor Service - Makefile
# =====================================

# Variables
COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := docker-compose -f $(COMPOSE_FILE)

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

# View logs
docker-logs:
	@$(DOCKER_COMPOSE) logs -f

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
	@sam build --template sam-template.yaml
	@echo "✅ Build complete"

# Cache dependencies for faster builds
sam-deps:
	@echo "📦 Caching dependencies for faster builds..."
	@mkdir -p .aws-sam/deps
	@uv pip install -r requirements-lambda.txt --target .aws-sam/deps/
	@echo "✅ Dependencies cached in .aws-sam/deps/"

# Build for local testing
sam-build-local:
	@echo "🔨 Building for local testing..."
	@sam build --template sam-template-local.yaml
	@echo "✅ Local build complete"

# Fast build using cached dependencies
sam-build-fast: sam-deps
	@echo "🔨 Fast building with cached dependencies..."
	@sam build --template sam-template-local.yaml --cached
	@echo "✅ Fast build complete"

sam-run-local:
	@echo "🚀 Starting SAM Local API..."
	@echo "📡 API available at http://localhost:3080"
	@echo "📝 Logs will be written to server.log"
	@sam local start-api \
		--port 3080 \
		--template .aws-sam/build/template.yaml \
		--log-file server.log \
		--skip-pull-image \
		--warm-containers EAGER \
		--parameter-overrides \
			Environment=local \
			DynamoDBEndpoint=http://localhost:8000

# Start local API (standard build)
sam-local: db-start sam-build-local sam-run-local
	@echo "✅ SAM Local API running"

# Start local API with fast build (using cached dependencies and layers)
sam-local-fast: db-start sam-build-fast sam-run-local
	@echo "✅ SAM Local API running (fast mode)"

# Clean dependency cache
sam-clean-deps:
	@echo "🧹 Cleaning dependency cache..."
	@rm -rf .aws-sam/deps
	@echo "✅ Dependency cache cleaned"

# Deploy to AWS (interactive)
sam-deploy: sam-build
	@echo "🚀 Deploying to AWS..."
	@sam deploy --guided

# Deploy with specific environment
sam-deploy-env: sam-build
	@if [ -z "$(ENV)" ]; then \
		echo "❌ Error: ENV parameter required. Usage: make sam-deploy-env ENV=dev"; \
		exit 1; \
	fi
	@echo "🚀 Deploying to $(ENV) environment..."
	@sam deploy --config-env $(ENV)

# Invoke Lambda function locally
sam-invoke:
	@if [ -z "$(FUNC)" ]; then \
		echo "❌ Error: FUNC parameter required. Usage: make sam-invoke FUNC=WeightProcessorFunction"; \
		exit 1; \
	fi
	@echo "⚡ Invoking $(FUNC)..."
	@sam local invoke $(FUNC) --event ../test_events/process_event.json

# View Lambda logs
sam-logs:
	@if [ -z "$(STACK)" ]; then \
		echo "📋 Viewing local SAM logs..."; \
		tail -f server.log 2>/dev/null || echo "No server.log found. Start API with: make sam-local"; \
	else \
		echo "📋 Viewing logs for stack $(STACK)..."; \
		sam logs -n WeightProcessorFunction --stack-name $(STACK) --tail; \
	fi

# View recent log entries (without following)
sam-logs-recent:
	@echo "📋 Recent log entries:"
	@tail -100 server.log 2>/dev/null || echo "No server.log found"

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
	@curl -s http://localhost:3080/health | jq '.' || echo "API not running. Start with: make sam-local"

# Test process endpoint
test-process:
	@echo "📊 Testing process endpoint..."
	@curl -X POST http://localhost:3080/process \
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
