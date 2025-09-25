# Testing SAM local without API key requirement

When running locally with SAM, API keys are typically not enforced. 
If you still get "Missing Authentication Token", try these approaches:

## Option 1: Direct curl without any auth
```bash
curl http://localhost:5448/api/v1/health
```

## Option 2: With a dummy API key header
```bash
curl http://localhost:5448/api/v1/health \
  -H "x-api-key: dummy-key-for-local"
```

## Option 3: Rebuild and restart
```bash
make docker-clean
make docker-run
# Then in another terminal:
make docker-health
```

## Option 4: Test directly with Lambda invoke
```bash
sam local invoke WeightProcessorFunction \
  --event test_events/health_check.json \
  --docker-network bridge
```

The health endpoint should work without authentication in local mode.
