# DynamoDB Local Docker Setup

## Quick Start

### Start DynamoDB Local
```bash
# Start DynamoDB and Admin UI
docker-compose up -d

# Or using make
make docker-up
```

Services will be available at:
- DynamoDB: http://localhost:8000
- DynamoDB Admin UI: http://localhost:8001

### Stop Services
```bash
# Stop containers
docker-compose down

# Or using make
make docker-down
```

### View DynamoDB Data
Open http://localhost:8001 in your browser to access the DynamoDB Admin UI.

## Common Commands

| Command | Description |
|---------|------------|
| `make docker-up` | Start DynamoDB services |
| `make docker-down` | Stop DynamoDB services |
| `make docker-restart` | Restart services |
| `make docker-status` | Check container status |
| `make docker-logs` | View container logs |
| `make docker-clean` | Remove containers and volumes |

## Using with SAM Local

When running SAM local, set the DynamoDB endpoint:
```bash
export DYNAMODB_ENDPOINT=http://localhost:8000
./scripts/start_local.sh
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process or change the port in docker-compose.yml
```

### Container Won't Start
```bash
# Check logs
docker-compose logs dynamodb-local

# Clean and restart
make docker-clean
make docker-up
```