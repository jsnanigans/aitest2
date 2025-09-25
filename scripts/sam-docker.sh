#!/bin/bash

# SAM Docker Helper Script
# This script provides convenient commands for running AWS SAM in a Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.sam.yml"
CONTAINER_NAME="weight-processor-sam"

# Helper functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Start the SAM environment
start_env() {
    print_info "Starting SAM Docker environment..."
    docker-compose -f $COMPOSE_FILE up -d

    print_info "Waiting for services to be ready..."
    sleep 5

    # Check if DynamoDB is ready
    for i in {1..30}; do
        if docker-compose -f $COMPOSE_FILE exec -T dynamodb-local curl -s http://localhost:8000 > /dev/null 2>&1; then
            print_info "✓ DynamoDB is ready"
            break
        fi
        echo -n "."
        sleep 1
    done

    print_info "SAM environment is ready!"
    print_info "DynamoDB Admin UI: http://localhost:8001"
}

# Stop the SAM environment
stop_env() {
    print_info "Stopping SAM Docker environment..."
    docker-compose -f $COMPOSE_FILE down
}

# Execute a command in the SAM container
exec_in_container() {
    docker-compose -f $COMPOSE_FILE exec sam-builder "$@"
}

# Run SAM build
sam_build() {
    print_info "Building SAM application..."
    exec_in_container bash -c "cd aws && sam build --template template.yaml"
}

# Run SAM local start-api
sam_local_api() {
    print_info "Starting SAM Local API..."
    print_info "API will be available at http://localhost:3000"

    # First build the application
    sam_build

    # Then start the API
    exec_in_container bash -c "cd aws && sam local start-api --host 0.0.0.0 --port 3000 --docker-network sam-network --container-host host.docker.internal"
}

# Run SAM local invoke
sam_invoke() {
    local function_name=$1
    local event_file=$2

    if [ -z "$function_name" ]; then
        print_error "Function name is required"
        echo "Usage: $0 invoke <function-name> [event-file]"
        exit 1
    fi

    print_info "Invoking function: $function_name"

    if [ -n "$event_file" ]; then
        exec_in_container bash -c "cd aws && sam local invoke $function_name --event $event_file --docker-network sam-network"
    else
        exec_in_container bash -c "cd aws && sam local invoke $function_name --docker-network sam-network"
    fi
}

# Initialize DynamoDB tables
init_db() {
    print_info "Initializing DynamoDB tables..."
    exec_in_container python scripts/init-dynamodb.py
}

# Run tests in the container
run_tests() {
    print_info "Running tests in SAM container..."
    exec_in_container python -m pytest tests/ -xvs
}

# Open a shell in the SAM container
shell() {
    print_info "Opening shell in SAM container..."
    docker-compose -f $COMPOSE_FILE exec sam-builder bash
}

# Show logs
logs() {
    docker-compose -f $COMPOSE_FILE logs -f sam-builder
}

# Show help
show_help() {
    echo "SAM Docker Helper Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start       - Start the SAM Docker environment"
    echo "  stop        - Stop the SAM Docker environment"
    echo "  build       - Build the SAM application"
    echo "  api         - Start SAM Local API (accessible at http://localhost:3000)"
    echo "  invoke      - Invoke a Lambda function locally"
    echo "  init-db     - Initialize DynamoDB tables"
    echo "  test        - Run tests in the container"
    echo "  shell       - Open a bash shell in the SAM container"
    echo "  logs        - Show container logs"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # Start the environment"
    echo "  $0 api                             # Start the local API"
    echo "  $0 invoke WeightProcessorFunction  # Invoke a function"
    echo "  $0 shell                           # Open a shell"
}

# Main script logic
check_docker

case "$1" in
    start)
        start_env
        ;;
    stop)
        stop_env
        ;;
    build)
        sam_build
        ;;
    api)
        sam_local_api
        ;;
    invoke)
        shift
        sam_invoke "$@"
        ;;
    init-db)
        init_db
        ;;
    test)
        run_tests
        ;;
    shell)
        shell
        ;;
    logs)
        logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac