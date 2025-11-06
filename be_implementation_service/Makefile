# Weight Processor Service

COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := docker-compose -f $(COMPOSE_FILE)

.PHONY: setup db-start db-stop db-reset build-local build-prod build-layer
.PHONY: sam-local test-api clean sam-list-failed sam-delete-stack
.PHONY: sam-delete-dev sam-delete-qa sam-delete-prod sam-info-dev
.PHONY: sam-build-prod sam-deploy-prod sam-deploy-dev sam-deploy-qa info

# Setup

setup:
	uv venv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

# Database

db-start:
	@docker ps | grep -q weight-processor-dynamodb || \
		($(DOCKER_COMPOSE) up -d dynamodb-local dynamodb-admin && sleep 3)

db-stop:
	@$(DOCKER_COMPOSE) stop dynamodb-local dynamodb-admin

db-reset: db-stop db-start
	@sleep 2
	@export DYNAMODB_ENDPOINT=http://localhost:8000 && \
	export DYNAMODB_TABLE_NAME=weight-processor-state && \
	export AWS_ACCESS_KEY_ID=local && \
	export AWS_SECRET_ACCESS_KEY=local && \
	uv run python scripts/init-dynamodb.py

# Build

build-local:
	@rm -rf build/local
	@mkdir -p build/local
	@cp -r src/ build/local/src/
	@cp config.toml build/local/
	@cp requirements.txt build/local/requirements.txt

build-prod:
	@rm -rf build/function
	@mkdir -p build/function
	@cp -r src/ build/function/src/
	@cp config.toml build/function/
	@echo "pydantic>=2.11.9" > build/function/requirements.txt
	@echo "aws-lambda-powertools>=2.37.0" >> build/function/requirements.txt

build-layer:
	@rm -rf build/layer
	@mkdir -p build/layer
	@cp requirements-layer.txt build/layer/requirements.txt

# Local Testing

sam-local: db-start build-local
	@sam build --template sam-template-local.yaml
	@AWS_ACCESS_KEY_ID=local \
	AWS_SECRET_ACCESS_KEY=local \
	AWS_DEFAULT_REGION=us-east-1 \
	sam local start-api \
		--port 3080 \
		--template .aws-sam/build/template.yaml \
		--log-file ../weight-processor-server.log \
		--skip-pull-image \
		--warm-containers EAGER \
		--parameter-overrides Environment=local DynamoDBEndpoint=http://localhost:8000

test-api:
	@curl -s http://localhost:3080/health | jq '.'

# Cleanup

clean:
	@rm -rf build/ .aws-sam __pycache__ .pytest_cache htmlcov .coverage weight-processor-server.log
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

# Stack Management

sam-list-failed:
	@aws cloudformation describe-stacks \
		--query "Stacks[?contains(StackName, 'WeightProcessor') && (StackStatus=='ROLLBACK_COMPLETE' || StackStatus=='ROLLBACK_IN_PROGRESS' || StackStatus=='DELETE_FAILED' || StackStatus=='CREATE_FAILED' || StackStatus=='UPDATE_ROLLBACK_COMPLETE')].{Name:StackName,Status:StackStatus}" \
		--output table 2>/dev/null || echo "No failed stacks"

sam-delete-stack:
	@[ -n "$(STACK_NAME)" ] || (echo "Error: STACK_NAME required" && exit 1)
	@STATUS=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND"); \
	if [ "$$STATUS" = "NOT_FOUND" ]; then \
		echo "Stack not found: $(STACK_NAME)" && exit 1; \
	elif echo "$$STATUS" | grep -qE "ROLLBACK|FAILED"; then \
		aws cloudformation delete-stack --stack-name $(STACK_NAME) && \
		aws cloudformation wait stack-delete-complete --stack-name $(STACK_NAME); \
	else \
		echo "Stack is healthy ($$STATUS). Use: aws cloudformation delete-stack --stack-name $(STACK_NAME)" && exit 1; \
	fi

sam-delete-dev:
	@STATUS=$$(aws cloudformation describe-stacks --stack-name WeightProcessorDevUs \
		--query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND"); \
	if [ "$$STATUS" = "NOT_FOUND" ]; then \
		echo "Stack not found" && exit 1; \
	elif echo "$$STATUS" | grep -qE "ROLLBACK|FAILED"; then \
		aws cloudformation delete-stack --stack-name WeightProcessorDevUs && \
		aws cloudformation wait stack-delete-complete --stack-name WeightProcessorDevUs; \
	elif [ "$(FORCE)" = "1" ]; then \
		aws cloudformation delete-stack --stack-name WeightProcessorDevUs && \
		aws cloudformation wait stack-delete-complete --stack-name WeightProcessorDevUs; \
	else \
		echo "Stack is healthy. Use: make sam-delete-dev FORCE=1" && exit 1; \
	fi

sam-delete-qa:
	@STATUS=$$(aws cloudformation describe-stacks --stack-name WeightProcessorQaUs \
		--query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND"); \
	if [ "$$STATUS" = "NOT_FOUND" ]; then \
		echo "Stack not found" && exit 1; \
	elif echo "$$STATUS" | grep -qE "ROLLBACK|FAILED"; then \
		aws cloudformation delete-stack --stack-name WeightProcessorQaUs && \
		aws cloudformation wait stack-delete-complete --stack-name WeightProcessorQaUs; \
	else \
		echo "Stack is healthy. Manual deletion required." && exit 1; \
	fi

sam-delete-prod:
	@echo "Production deletion disabled. Use AWS Console." && exit 1

sam-info-dev:
	@aws cloudformation describe-stacks --stack-name WeightProcessorDevUs \
		--query 'Stacks[0].{Status:StackStatus,Created:CreationTime,Updated:LastUpdatedTime}' \
		--output table
	@aws cloudformation describe-stacks --stack-name WeightProcessorDevUs \
		--query 'Stacks[0].Outputs' --output table
	@aws cloudformation describe-stack-resources --stack-name WeightProcessorDevUs \
		--query 'StackResources[].{Type:ResourceType,LogicalId:LogicalResourceId,PhysicalId:PhysicalResourceId,Status:ResourceStatus}' \
		--output table

# Deployment

sam-build-prod: build-prod build-layer
	@sam build --template sam-template-prod.yaml

sam-deploy-prod: sam-build-prod
	@sam deploy --config-env default

sam-deploy-dev: sam-build-prod
	@sam deploy --config-env dev

sam-deploy-qa: sam-build-prod
	@sam deploy --config-env qa

# Info

info:
	@python --version
	@docker --version
	@docker-compose --version
	@sam --version 2>/dev/null || echo "SAM CLI not installed"
	@uv --version 2>/dev/null || echo "uv not installed"
