.PHONY: help install test test-lambda build deploy local clean

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run all tests"
	@echo "  make test-lambda   Run Lambda handler tests"
	@echo "  make build         Build Lambda package"
	@echo "  make deploy        Deploy to AWS (dev)"
	@echo "  make deploy-staging Deploy to staging"
	@echo "  make deploy-prod   Deploy to production"
	@echo "  make local         Run Lambda locally"
	@echo "  make clean         Clean build artifacts"

install:
	uv pip install -r requirements.txt
	uv pip install -r requirements-lambda.txt

test:
	uv run python -m pytest tests/ -xvs

test-lambda:
	uv run python -m pytest tests/test_lambda_handler.py -xvs

build:
	sam build --use-container

deploy:
	./scripts/deploy/deploy.sh --env dev

deploy-staging:
	./scripts/deploy/deploy.sh --env staging

deploy-prod:
	./scripts/deploy/deploy.sh --env prod --auto-confirm

local:
	./scripts/deploy/test_local.sh

clean:
	rm -rf .aws-sam
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete