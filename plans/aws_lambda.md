# AWS Lambda API Plan and Design

## Objectives
- Accept 1..N readings via HTTP API.
- Each reading includes `user_id`, `id`, `effectiveDateTime`, and measurement fields.
- Return per-reading results: quality score, accepted/rejected, reason.
- Enforce per-user monotonic timestamps: reject if `effectiveDateTime` < last accepted timestamp.
- Ensure idempotency on `(user_id, id)` across retries and batches.
- Preserve core architecture: `processor.py` stateless; persistence in `processor_database.py`.

## Endpoints
- POST `/readings` (API Gateway → Lambda)
  - Sync processing. Partial failures allowed (multi-status semantics).
- Optional (ops): GET `/users/{user_id}/state` to inspect `last_timestamp` for diagnostics.

## Request Schema (JSON)
- Top-level
  - `request_id` string optional (trace/idempotency; default to API GW `requestId`).
  - `items` array required, 1..`BATCH_MAX`.
- Item required fields
  - `user_id` string
  - `id` string (unique per `user_id`)
  - `effectiveDateTime` string (ISO 8601; UTC recommended)
  - `measurement` object (domain-specific; minimally `{ value: number, unit: string }`)
  - `options` object optional (processor knobs)
- Example
```json
{
  "request_id": "req-123",
  "items": [
    {
      "user_id": "u-1",
      "id": "m-1001",
      "effectiveDateTime": "2025-09-15T10:45:00Z",
      "measurement": { "value": 78.2, "unit": "kg" },
      "options": { "smooth": true }
    }
  ]
}
```

## Response Schema (JSON)
- Top-level
  - `status`: "ok" | "partial" | "error"
  - `summary`: counts `{ received, processed, accepted, rejected, duplicates, errors }`
  - `results`: array per input item (same order unless `RESPONSE_PRESERVE_ORDER=false`)
- Per-item result
  - `user_id`, `id`
  - `accepted`: boolean
  - `quality_score`: number|null
  - `reason`: null|"duplicate"|"timestamp_conflict"|"validation_error"|"processing_error"
  - `message`: short human message
  - `effectiveDateTime`: echo
- HTTP status
  - 200: all accepted
  - 207: partial acceptance/rejections
  - 400: all invalid
  - 413: payload too large
  - 429: throttled
  - 500: all failed due to server error
- Example (partial)
```json
{
  "status": "partial",
  "summary": { "received": 3, "processed": 3, "accepted": 2, "rejected": 1, "duplicates": 0, "errors": 1 },
  "results": [
    { "user_id": "u-1", "id": "m-1001", "accepted": true, "quality_score": 0.92, "reason": null, "message": null, "effectiveDateTime": "2025-09-15T10:45:00Z" },
    { "user_id": "u-1", "id": "m-1000", "accepted": false, "quality_score": null, "reason": "timestamp_conflict", "message": "effectiveDateTime older than last accepted (2025-09-15T10:45:00Z)", "effectiveDateTime": "2025-09-15T10:40:00Z" },
    { "user_id": "u-2", "id": "m-5", "accepted": true, "quality_score": 0.88, "reason": null, "message": null, "effectiveDateTime": "2025-09-15T11:00:00Z" }
  ]
}
```

## Validation Rules
- Syntactic: JSON parse, required fields, ISO datetime, numeric value within domain bounds.
- Semantic:
  - Idempotency: `(user_id, id)` must be unique; if exists with same payload → return existing result; if exists with different payload → item-level conflict (duplicate).
  - Monotonicity: `effectiveDateTime` must be >= last accepted timestamp for `user_id`; otherwise reject that item with `timestamp_conflict`.
- Limits: `items.length <= BATCH_MAX` (e.g., 500), body size <= `MAX_PAYLOAD_BYTES`.

## Processing Flow (Lambda Handler)
- Parse body; derive `correlation_id` from `request_id` or API Gateway `requestContext.requestId`.
- Group items by `user_id`; within each group, sort ascending by `effectiveDateTime`.
- For each `user_id` group:
  - Read user state: `last_timestamp` + `version` from DB.
  - For each item in order:
    - If `item.effectiveDateTime` < `last_timestamp` → mark `timestamp_conflict`; continue.
    - Idempotency check: lookup `(user_id, id)`:
      - If found and payload hash matches → reuse stored result (duplicate same-payload).
      - If found and payload differs → mark `duplicate` conflict.
      - If not found → compute quality score via stateless processor.
    - Persist via conditional write (transaction):
      - Put Reading if not exists.
      - Update UserState `last_timestamp = max(last_timestamp, item.effectiveDateTime)` with condition on `version`.
    - On conditional failure (race): re-read state and re-evaluate; if `effectiveDateTime` now older → mark `timestamp_conflict`.
- Aggregate per-item results; set top-level HTTP code (200/207/...).

## Persistence Design (DynamoDB, Single-Table)
- Table: `readings`
  - PK: `PK` string, SK: `SK` string
- Entities
  - UserState: `PK="USER#<user_id>"`, `SK="STATE"`, attrs: `last_timestamp` (ISO string), `version` (number), `updated_at`.
  - Reading: `PK="USER#<user_id>"`, `SK="TS#<epoch_ms>#ID#<id>"`, attrs: `id`, `effectiveDateTime`, `payload_hash`, `quality_score`, `accepted` (bool), `reason` (nullable), `created_at`.
- GSIs
  - `GSI1` (Idempotency): `GSI1PK="IDEMP#<user_id>"`, `GSI1SK="<id>"` on Reading items to find by `(user_id, id)` quickly.
  - Optional `GSI2` (Recent): `GSI2PK="USER#<user_id>"`, `GSI2SK="UPDATED#<updated_at>"` to query recents.
- Conditional logic
  - Initialize UserState lazily with `last_timestamp = "1970-01-01T00:00:00Z"`, `version=0`.
  - For accept: TransactWriteItems:
    - ConditionCheck UserState `version = expected_version` and `last_timestamp <= item.effectiveDateTime`.
    - Put Reading with condition `attribute_not_exists(PK)` (or not exists GSI1).
    - Update UserState set `last_timestamp = item.effectiveDateTime`, `version = version + 1`.
- TTL
  - Optional TTL on rejected items if retention not required.

## Idempotency
- Key: `(user_id, id)` via `GSI1`; `payload_hash = sha256(item without id/request_id)`.
- Behavior:
  - Same payload: return stored result (no-op).
  - Different payload: item-level conflict (`reason: "duplicate"`, 409 in result).

## Processor Integration
- Keep `processor.py` stateless; map API items to `load → process → save` per item.
- Implement a DynamoDB-backed `processor_database.py` adapter with the same interface as in-memory.
- Maintain per-measurement transaction boundaries; no cross-item state.

## Packaging & Runtime
- Choose one:
  - Lambda Layer + zip: move numeric libs (`numpy`, `pykalman`, `matplotlib`) to a manylinux2014-compatible Layer; function zip stays thin.
  - Container image: base `public.ecr.aws/lambda/python:3.11` with all deps; simpler for heavy binaries.
- Set `MPLBACKEND=Agg` if visualization paths exist; make plotting optional.
- Memory 1024–2048 MB; timeout sized for batch size; ephemeral storage if needed.

## Infrastructure as Code (SAM/Serverless)
- Resources
  - API: `POST /readings` (and optional `GET /users/{user_id}/state`).
  - Lambda function (Python 3.11) with env vars and reserved concurrency if needed.
  - DynamoDB table + GSIs and autoscaling or on-demand mode.
  - IAM role: scoped access to DDB, CloudWatch Logs, X-Ray.
  - Optional DLQ for future async patterns; CloudWatch Alarms and Dashboards.
- Stages
  - `dev`, `staging`, `prod` with distinct tables or name prefixes.

## Configuration (Env Vars)
- `DB_TABLE=readings`
- `STATE_BACKEND=dynamodb`
- `BATCH_MAX=500`, `MAX_PAYLOAD_BYTES=2000000`
- `LOG_LEVEL=info`, `RESPONSE_PRESERVE_ORDER=true`
- `FAIL_FAST=false`, `REQUIRE_IDEMPOTENCY=true`
- `MPLBACKEND=Agg`

## Error Mapping (per-item)
- `validation_error`: missing fields, bad types, out-of-range values.
- `duplicate`: `(user_id,id)` exists with different payload.
- `timestamp_conflict`: `effectiveDateTime` < current `last_timestamp` at commit time.
- `processing_error`: exceptions in processor.

## Security
- AuthN: JWT (Cognito Authorizer) or IAM SigV4.
- AuthZ: enforce `user_id` equals token subject unless elevated scope provided.
- WAF: rate-based rules and size constraints.
- Throttling: API GW stage throttles per route; optional per-user rate limiting via DDB token bucket.

## Observability
- Logging: structured JSON including `correlation_id`, `user_id`, `id`, decision, duration.
- Metrics: AWS Embedded Metrics Format for `items_received`, `accepted`, `rejected`, `duplicates`, `timestamp_conflicts`, `latency_ms`.
- Tracing: X-Ray for API and Lambda; annotate `user_id`, batch size, outcomes.
- Alarms: Lambda errors > 0, p95 latency near timeout, throttle counts; DDB throttles.

## Performance & Limits
- Batch: per `user_id` sequential processing to maintain order; process users concurrently.
- Lambda sizing: 1024–2048 MB; timeout 10–30s depending on `BATCH_MAX`.
- DDB autoscaling: target utilization ~70% RCU/WCU or on-demand for simplicity.
- Cold start: consider container image if numeric stack is heavy; otherwise Layer.

## Testing Plan
- Unit: schema validation; idempotency; timestamp rules; batch ordering per user; per-item error mapping.
- Integration: LocalStack DDB; `sam local start-api`; race tests for conditional updates.
- Property: random timestamp sequences to assert monotonic enforcement.
- Load: worst-case `BATCH_MAX` across many `user_id`s; measure DDB capacity and latency.

## Rollout
- Deploy `dev` stack; run synthetic batches with mixed ordering; verify DDB contents and `last_timestamp`.
- Tune memory/timeouts; set autoscaling; add WAF rules and throttles.
- Promote to `staging`, then `prod` with alarms and dashboards enabled.

