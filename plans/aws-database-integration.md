# AWS Weight Processor - Database & Integration Guide

## 1. DynamoDB Client Implementation

### 1.1 Complete DynamoDB Client
```python
# src/database/dynamodb_client.py
import boto3
from typing import Dict, Any, Optional, List
from datetime import datetime
from botocore.exceptions import ClientError
from decimal import Decimal
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)

class DynamoDBClient:
    """
    Low-level DynamoDB client for weight processor state management.
    Handles all database operations with proper error handling and retries.
    """

    def __init__(self, table_name: str = None, region: str = 'us-east-1'):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table_name = table_name or os.environ.get('STATE_TABLE_NAME')
        self.table = self.dynamodb.Table(self.table_name)

    def get_item(self, Key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get single item from DynamoDB."""
        try:
            response = self.table.get_item(Key=Key)
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Error getting item: {e}")
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                raise ValueError(f"Table {self.table_name} not found")
            raise

    def put_item(self, Item: Dict[str, Any], **kwargs) -> bool:
        """Put item into DynamoDB with optional conditions."""
        try:
            # Convert floats to Decimal for DynamoDB
            item = self._convert_floats_to_decimal(Item)
            self.table.put_item(Item=item, **kwargs)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logger.warning("Conditional check failed")
                return False
            logger.error(f"Error putting item: {e}")
            raise

    def delete_item(self, Key: Dict[str, Any]) -> bool:
        """Delete item from DynamoDB."""
        try:
            self.table.delete_item(Key=Key)
            return True
        except ClientError as e:
            logger.error(f"Error deleting item: {e}")
            return False

    def query(self, **kwargs) -> Dict[str, Any]:
        """Query items from DynamoDB."""
        try:
            response = self.table.query(**kwargs)
            # Convert Decimal back to float
            if 'Items' in response:
                response['Items'] = [
                    self._convert_decimal_to_float(item)
                    for item in response['Items']
                ]
            return response
        except ClientError as e:
            logger.error(f"Error querying items: {e}")
            raise

    def batch_write(self, items: List[Dict[str, Any]]) -> bool:
        """Batch write items to DynamoDB."""
        try:
            with self.table.batch_writer() as batch:
                for item in items:
                    item = self._convert_floats_to_decimal(item)
                    batch.put_item(Item=item)
            return True
        except ClientError as e:
            logger.error(f"Error in batch write: {e}")
            return False

    def _convert_floats_to_decimal(self, obj):
        """Convert float values to Decimal for DynamoDB."""
        if isinstance(obj, list):
            return [self._convert_floats_to_decimal(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            return Decimal(str(obj))
        else:
            return obj

    def _convert_decimal_to_float(self, obj):
        """Convert Decimal values back to float."""
        if isinstance(obj, list):
            return [self._convert_decimal_to_float(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, Decimal):
            return float(obj)
        else:
            return obj
```

### 1.2 Repository Pattern Implementation
```python
# src/database/repository.py
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .dynamodb_client import DynamoDBClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

class StateRepository:
    """
    Repository for managing weight processor states.
    Provides high-level operations on top of DynamoDB.
    """

    def __init__(self, table_name: str = None):
        self.client = DynamoDBClient(table_name)
        self.snapshot_prefix = "snapshot_"
        self.current_state_key = "current"

    def get_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a user."""
        return self.client.get_item(
            Key={
                'userId': user_id,
                'stateType': self.current_state_key
            }
        )

    def save_user_state(self, user_id: str, state: Dict[str, Any],
                        version_check: bool = True) -> bool:
        """Save user state with optional optimistic locking."""
        state['userId'] = user_id
        state['stateType'] = self.current_state_key
        state['updatedAt'] = datetime.utcnow().isoformat()

        if version_check and 'version' in state:
            return self.client.put_item(
                Item=state,
                ConditionExpression='attribute_not_exists(userId) OR version < :v',
                ExpressionAttributeValues={':v': state['version']}
            )
        else:
            return self.client.put_item(Item=state)

    def get_user_snapshots(self, user_id: str,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent snapshots for a user."""
        response = self.client.query(
            KeyConditionExpression='userId = :uid AND begins_with(stateType, :prefix)',
            ExpressionAttributeValues={
                ':uid': user_id,
                ':prefix': self.snapshot_prefix
            },
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )
        return response.get('Items', [])

    def get_snapshot_at_time(self, user_id: str,
                             timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot before a given timestamp."""
        response = self.client.query(
            KeyConditionExpression='userId = :uid AND stateType < :st',
            ExpressionAttributeValues={
                ':uid': user_id,
                ':st': f"{self.snapshot_prefix}{timestamp.isoformat()}"
            },
            ScanIndexForward=False,
            Limit=1
        )

        items = response.get('Items', [])
        return items[0] if items else None

    def create_snapshot(self, user_id: str, state: Dict[str, Any],
                       retention_days: int = 7) -> bool:
        """Create a snapshot with TTL."""
        timestamp = datetime.utcnow()
        snapshot = state.copy()
        snapshot.update({
            'userId': user_id,
            'stateType': f"{self.snapshot_prefix}{timestamp.isoformat()}",
            'snapshotTime': timestamp.isoformat(),
            'ttl': int((timestamp + timedelta(days=retention_days)).timestamp())
        })
        return self.client.put_item(Item=snapshot)

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user."""
        # Delete current state
        self.client.delete_item(
            Key={'userId': user_id, 'stateType': self.current_state_key}
        )

        # Delete all snapshots
        snapshots = self.get_user_snapshots(user_id, limit=100)
        for snapshot in snapshots:
            self.client.delete_item(
                Key={
                    'userId': user_id,
                    'stateType': snapshot['stateType']
                }
            )

        return True

    def batch_get_states(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get states for multiple users efficiently."""
        keys = [
            {'userId': uid, 'stateType': self.current_state_key}
            for uid in user_ids
        ]

        response = self.client.dynamodb.batch_get_item(
            RequestItems={
                self.client.table_name: {
                    'Keys': keys
                }
            }
        )

        states = {}
        for item in response.get('Responses', {}).get(self.client.table_name, []):
            states[item['userId']] = item

        return states
```

## 2. Java Backend Integration

### 2.1 Complete Java Client Library
```java
// WeightProcessorClient.java
package com.example.weightprocessor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class WeightProcessorClient {
    private static final Logger logger = LoggerFactory.getLogger(WeightProcessorClient.class);

    private final String apiEndpoint;
    private final String apiKey;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final int maxRetries;
    private final Duration requestTimeout;

    public WeightProcessorClient(WeightProcessorConfig config) {
        this.apiEndpoint = config.getApiEndpoint();
        this.apiKey = config.getApiKey();
        this.maxRetries = config.getMaxRetries();
        this.requestTimeout = config.getRequestTimeout();

        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(requestTimeout.toMillis(), TimeUnit.MILLISECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .addInterceptor(new RetryInterceptor(maxRetries))
            .addInterceptor(new LoggingInterceptor())
            .build();

        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
    }

    /**
     * Process all historical data for a user (one-time cleanup)
     */
    public CleanupResponse cleanup(String userId, CleanupRequest request)
            throws WeightProcessorException {
        String url = String.format("%s/api/v1/cleanup/%s", apiEndpoint, userId);

        try {
            String jsonBody = objectMapper.writeValueAsString(request);

            Request httpRequest = new Request.Builder()
                .url(url)
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("X-Correlation-Id", UUID.randomUUID().toString())
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .build();

            Response response = executeWithRetry(httpRequest);

            if (!response.isSuccessful()) {
                handleErrorResponse(response);
            }

            return objectMapper.readValue(
                response.body().string(),
                CleanupResponse.class
            );

        } catch (IOException e) {
            logger.error("Failed to process cleanup for user: {}", userId, e);
            throw new WeightProcessorException("Cleanup failed", e);
        }
    }

    /**
     * Process multiple new measurements
     * All measurements must be after the last processed timestamp
     */
    public ProcessResponse process(String userId, List<Measurement> measurements)
            throws WeightProcessorException, HistoricalConflictException {
        String url = String.format("%s/api/v1/process/%s", apiEndpoint, userId);

        try {
            ProcessRequest request = new ProcessRequest(measurements);
            String jsonBody = objectMapper.writeValueAsString(request);

            Request httpRequest = new Request.Builder()
                .url(url)
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("X-Correlation-Id", UUID.randomUUID().toString())
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .build();

            Response response = executeWithRetry(httpRequest);

            // Check for historical conflict
            if (response.code() == 409) {
                String responseBody = response.body().string();
                HistoricalConflictResponse conflict = objectMapper.readValue(
                    responseBody,
                    HistoricalConflictResponse.class
                );
                throw new HistoricalConflictException(conflict);
            }

            if (!response.isSuccessful()) {
                handleErrorResponse(response);
            }

            return objectMapper.readValue(
                response.body().string(),
                ProcessResponse.class
            );

        } catch (IOException e) {
            logger.error("Failed to process measurements for user: {}", userId, e);
            throw new WeightProcessorException("Process failed", e);
        }
    }

    /**
     * Process a single measurement (convenience method)
     */
    public ProcessResponse processSingle(String userId, Measurement measurement)
            throws WeightProcessorException, HistoricalConflictException {
        return process(userId, Arrays.asList(measurement));
    }

    /**
     * Replay historical measurements from a specific point
     */
    public ReplayResponse replay(String userId, ReplayRequest request)
            throws WeightProcessorException {
        String url = String.format("%s/api/v1/replay/%s", apiEndpoint, userId);

        try {
            String jsonBody = objectMapper.writeValueAsString(request);

            Request httpRequest = new Request.Builder()
                .url(url)
                .header("X-API-Key", apiKey)
                .header("Content-Type", "application/json")
                .header("X-Correlation-Id", UUID.randomUUID().toString())
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .build();

            Response response = executeWithRetry(httpRequest);

            if (!response.isSuccessful()) {
                handleErrorResponse(response);
            }

            return objectMapper.readValue(
                response.body().string(),
                ReplayResponse.class
            );

        } catch (IOException e) {
            logger.error("Failed to replay measurements for user: {}", userId, e);
            throw new WeightProcessorException("Replay failed", e);
        }
    }

    /**
     * Async version of cleanup
     */
    public CompletableFuture<CleanupResponse> cleanupAsync(String userId,
                                                           CleanupRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return cleanup(userId, request);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Process with automatic replay on conflict
     */
    public ProcessResponse processWithAutoReplay(String userId,
                                                 List<Measurement> measurements,
                                                 List<Measurement> historicalBuffer)
            throws WeightProcessorException {
        try {
            // Try normal processing first
            return process(userId, measurements);

        } catch (HistoricalConflictException e) {
            logger.info("Historical conflict detected, performing replay for user: {}",
                       userId);

            // Prepare replay request with both historical buffer and new measurements
            ReplayRequest replayRequest = new ReplayRequest();
            replayRequest.setReplayFromTimestamp(e.getConflict().getReplayFromTimestamp());

            // Combine historical buffer and new measurements
            List<Measurement> allMeasurements = new ArrayList<>();
            allMeasurements.addAll(historicalBuffer);
            allMeasurements.addAll(measurements);
            replayRequest.setMeasurements(allMeasurements);

            // Perform replay
            ReplayResponse replayResponse = replay(userId, replayRequest);

            // Convert replay response to process response format
            return convertReplayToProcessResponse(measurements, replayResponse);
        }
    }

    /**
     * Process single measurement with auto-replay (convenience method)
     */
    public ProcessResponse processSingleWithAutoReplay(String userId,
                                                       Measurement measurement,
                                                       List<Measurement> historicalBuffer)
            throws WeightProcessorException {
        return processWithAutoReplay(userId, Arrays.asList(measurement), historicalBuffer);
    }

    private Response executeWithRetry(Request request) throws IOException {
        return httpClient.newCall(request).execute();
    }

    private void handleErrorResponse(Response response) throws WeightProcessorException {
        try {
            String errorBody = response.body().string();
            ErrorResponse error = objectMapper.readValue(errorBody, ErrorResponse.class);

            throw new WeightProcessorException(
                String.format("API Error %d: %s", response.code(), error.getMessage())
            );
        } catch (IOException e) {
            throw new WeightProcessorException(
                String.format("API Error %d", response.code())
            );
        }
    }

    private ProcessResponse convertReplayToProcessResponse(List<Measurement> measurements,
                                                          ReplayResponse replay) {
        // Convert replay response to process response format
        ProcessResponse response = new ProcessResponse();
        response.setStatus("processed");

        // Extract results for our specific measurements
        Set<UUID> measurementIds = measurements.stream()
            .map(Measurement::getUuid)
            .collect(Collectors.toSet());

        List<MeasurementResult> relevantResults = replay.getMeasurements().stream()
            .filter(r -> measurementIds.contains(r.getUuid()))
            .collect(Collectors.toList());

        response.setMeasurements(relevantResults);
        response.setProcessedCount(relevantResults.size());
        response.setAcceptedCount((int) relevantResults.stream()
            .filter(MeasurementResult::isAccepted)
            .count());
        response.setRejectedCount((int) relevantResults.stream()
            .filter(r -> !r.isAccepted())
            .count());

        return response;
    }

    /**
     * Retry interceptor for automatic retries
     */
    private static class RetryInterceptor implements Interceptor {
        private final int maxRetries;

        public RetryInterceptor(int maxRetries) {
            this.maxRetries = maxRetries;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            Response response = null;
            IOException lastException = null;

            for (int i = 0; i <= maxRetries; i++) {
                try {
                    if (response != null) {
                        response.close();
                    }
                    response = chain.proceed(request);

                    // Don't retry on success or client errors
                    if (response.isSuccessful() ||
                        (response.code() >= 400 && response.code() < 500)) {
                        return response;
                    }

                } catch (IOException e) {
                    lastException = e;
                    if (i == maxRetries) {
                        throw e;
                    }
                }

                // Exponential backoff
                try {
                    long delay = (long) Math.pow(2, i) * 1000;
                    Thread.sleep(Math.min(delay, 10000));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted during retry", e);
                }
            }

            if (lastException != null) {
                throw lastException;
            }
            return response;
        }
    }

    /**
     * Logging interceptor for debugging
     */
    private static class LoggingInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();

            long startTime = System.nanoTime();
            logger.debug("Sending request: {} {}", request.method(), request.url());

            Response response = chain.proceed(request);

            long duration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startTime);
            logger.debug("Received response: {} {} in {}ms",
                        response.code(), request.url(), duration);

            return response;
        }
    }
}
```

### 2.2 Java Models
```java
// Measurement.java
package com.example.weightprocessor.models;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.Builder;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;

@Data
@Builder
public class Measurement {
    private UUID uuid;
    private double weight;
    private String unit;

    @JsonProperty("effectiveDateTime")
    private Instant effectiveDateTime;

    private String source;
    private Map<String, Object> metadata;
}

// CleanupRequest.java
@Data
@Builder
public class CleanupRequest {
    private List<Measurement> measurements;
    private UserProfile userProfile;
    private CleanupOptions options;

    @Data
    @Builder
    public static class CleanupOptions {
        @Builder.Default
        private boolean resetState = true;

        @Builder.Default
        private boolean includeQualityScores = true;

        @Builder.Default
        private boolean includeDebugInfo = false;
    }
}

// ProcessRequest.java
@Data
@Builder
public class ProcessRequest {
    private List<Measurement> measurements;
    private Map<String, Object> options;

    @Builder.Default
    private boolean failOnHistoricalConflict = true;
}

// ProcessResponse.java
@Data
public class ProcessResponse {
    private String status;
    private int processedCount;
    private int acceptedCount;
    private int rejectedCount;
    private List<MeasurementResult> measurements;
    private StateUpdate stateUpdate;

    @Data
    public static class StateUpdate {
        private Double previousWeight;
        private Double currentWeight;
        private Instant lastProcessedTimestamp;
    }
}

// CleanupResponse.java
@Data
public class CleanupResponse {
    private String userId;
    private int processedCount;
    private int acceptedCount;
    private int rejectedCount;
    private List<MeasurementResult> measurements;
    private FinalState finalState;

    @Data
    public static class MeasurementResult {
        private UUID uuid;
        private boolean accepted;
        private Double qualityScore;
        private Double kalmanEstimate;
        private Double kalmanUncertainty;
        private String rejectionReason;
        private String stage;
        private Map<String, Double> components;
    }

    @Data
    public static class FinalState {
        private double currentWeight;
        private double uncertainty;
        private Instant lastProcessedTimestamp;
        private int totalMeasurements;
        private String adaptationState;
    }
}

// HistoricalConflictException.java
public class HistoricalConflictException extends Exception {
    private final HistoricalConflictResponse conflict;

    public HistoricalConflictException(HistoricalConflictResponse conflict) {
        super(conflict.getError());
        this.conflict = conflict;
    }

    public HistoricalConflictResponse getConflict() {
        return conflict;
    }

    public Instant getReplayFromTimestamp() {
        return conflict.getDetails().getReplayFromTimestamp();
    }
}
```

### 2.3 Spring Boot Integration Example
```java
// WeightProcessorService.java
@Service
@Slf4j
public class WeightProcessorService {

    private final WeightProcessorClient client;
    private final WeightMeasurementRepository repository;
    private final CircuitBreaker circuitBreaker;

    @Autowired
    public WeightProcessorService(
            WeightProcessorClient client,
            WeightMeasurementRepository repository) {
        this.client = client;
        this.repository = repository;
        this.circuitBreaker = CircuitBreaker.ofDefaults("weight-processor");
    }

    /**
     * Process new weight measurements with automatic conflict handling
     */
    @Transactional
    public ProcessingResult processNewMeasurements(String userId,
                                                   List<WeightMeasurementEntity> entities) {
        try {
            // Convert entities to API models
            List<Measurement> measurements = entities.stream()
                .map(this::convertToMeasurement)
                .collect(Collectors.toList());

            // Get historical buffer in case replay is needed
            List<WeightMeasurementEntity> historicalEntities =
                repository.findRecentByUserId(userId, 100);
            List<Measurement> historicalBuffer =
                historicalEntities.stream()
                    .map(this::convertToMeasurement)
                    .collect(Collectors.toList());

            // Process with automatic replay
            ProcessResponse response = circuitBreaker.executeSupplier(() ->
                client.processWithAutoReplay(userId, measurements, historicalBuffer)
            );

            // Update entities with results
            Map<UUID, WeightMeasurementEntity> entityMap = entities.stream()
                .collect(Collectors.toMap(
                    WeightMeasurementEntity::getUuid,
                    Function.identity()
                ));

            for (ProcessResponse.MeasurementResult result : response.getMeasurements()) {
                WeightMeasurementEntity entity = entityMap.get(result.getUuid());
                if (entity != null) {
                    entity.setAccepted(result.isAccepted());
                    entity.setQualityScore(result.getQualityScore());
                    entity.setKalmanEstimate(result.getKalmanEstimate());
                    entity.setProcessedAt(Instant.now());
                }
            }

            repository.saveAll(entities);

            return ProcessingResult.success(entities);

        } catch (Exception e) {
            log.error("Failed to process measurements for user: {}", userId, e);
            return ProcessingResult.failure(e.getMessage());
        }
    }

    /**
     * Process single measurement (convenience method)
     */
    @Transactional
    public ProcessingResult processNewMeasurement(String userId,
                                                  WeightMeasurementEntity entity) {
        return processNewMeasurements(userId, Arrays.asList(entity));
    }

    /**
     * Perform one-time cleanup for a user
     */
    @Async
    public CompletableFuture<CleanupResult> performCleanup(String userId) {
        try {
            // Get all measurements for user
            List<WeightMeasurementEntity> entities =
                repository.findAllByUserId(userId);

            // Convert to API models
            List<Measurement> measurements = entities.stream()
                .map(this::convertToMeasurement)
                .collect(Collectors.toList());

            // Build request
            CleanupRequest request = CleanupRequest.builder()
                .measurements(measurements)
                .options(CleanupRequest.CleanupOptions.builder()
                    .resetState(true)
                    .includeQualityScores(true)
                    .build())
                .build();

            // Execute cleanup
            CleanupResponse response = client.cleanup(userId, request);

            // Update database with results
            updateEntitiesWithCleanupResults(entities, response);

            return CompletableFuture.completedFuture(
                CleanupResult.success(response)
            );

        } catch (Exception e) {
            log.error("Cleanup failed for user: {}", userId, e);
            return CompletableFuture.completedFuture(
                CleanupResult.failure(e.getMessage())
            );
        }
    }

    private Measurement convertToMeasurement(WeightMeasurementEntity entity) {
        return Measurement.builder()
            .uuid(entity.getUuid())
            .weight(entity.getWeight())
            .unit(entity.getUnit())
            .effectiveDateTime(entity.getEffectiveDateTime())
            .source(entity.getSource())
            .metadata(entity.getMetadata())
            .build();
    }

    @Transactional
    private void updateEntitiesWithCleanupResults(
            List<WeightMeasurementEntity> entities,
            CleanupResponse response) {

        Map<UUID, WeightMeasurementEntity> entityMap = entities.stream()
            .collect(Collectors.toMap(
                WeightMeasurementEntity::getUuid,
                Function.identity()
            ));

        for (CleanupResponse.MeasurementResult result : response.getMeasurements()) {
            WeightMeasurementEntity entity = entityMap.get(result.getUuid());
            if (entity != null) {
                entity.setAccepted(result.isAccepted());
                entity.setQualityScore(result.getQualityScore());
                entity.setKalmanEstimate(result.getKalmanEstimate());
                entity.setRejectionReason(result.getRejectionReason());
                entity.setProcessedAt(Instant.now());
            }
        }

        repository.saveAll(entities);
    }
}
```

## 3. Performance Optimization

### 3.1 Caching Layer
```python
# src/services/cache_service.py
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from functools import wraps

from ..utils.logger import get_logger

logger = get_logger(__name__)

class CacheService:
    """
    Redis-based caching for frequently accessed states.
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.client = redis.from_url(self.redis_url)
        self.default_ttl = 300  # 5 minutes

    def get_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached state."""
        key = f"state:{user_id}"
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    def set_state(self, user_id: str, state: Dict[str, Any],
                  ttl: int = None) -> bool:
        """Cache state with TTL."""
        key = f"state:{user_id}"
        ttl = ttl or self.default_ttl
        try:
            self.client.setex(
                key,
                ttl,
                json.dumps(state, default=str)
            )
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
        return False

    def invalidate(self, user_id: str) -> bool:
        """Invalidate cached state."""
        key = f"state:{user_id}"
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache invalidate failed: {e}")
        return False

def with_cache(ttl: int = 300):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, user_id: str, *args, **kwargs):
            # Try cache first
            cache_key = f"{func.__name__}:{user_id}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return cached

            # Call function
            result = func(self, user_id, *args, **kwargs)

            # Cache result
            if result:
                self.cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
```

### 3.2 Connection Pooling
```python
# src/database/connection_pool.py
from typing import Dict, Any
import boto3
from botocore.config import Config

class DynamoDBConnectionPool:
    """
    Connection pooling for DynamoDB clients.
    """

    _instances: Dict[str, Any] = {}

    @classmethod
    def get_client(cls, region: str = 'us-east-1'):
        """Get or create DynamoDB client with connection pooling."""
        if region not in cls._instances:
            config = Config(
                region_name=region,
                max_pool_connections=50,
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                }
            )
            cls._instances[region] = boto3.client('dynamodb', config=config)
        return cls._instances[region]
```

## 4. Monitoring & Metrics

### 4.1 Custom CloudWatch Metrics
```python
# src/utils/metrics.py
import boto3
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

@dataclass
class MetricData:
    name: str
    value: float
    unit: str = 'None'
    dimensions: Dict[str, str] = None

class MetricsCollector:
    """
    Collect and publish custom metrics to CloudWatch.
    """

    def __init__(self, namespace: str = 'WeightProcessor'):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = namespace
        self.buffer: List[MetricData] = []

    def record_processing_time(self, user_id: str, duration_ms: float):
        """Record processing latency."""
        self.buffer.append(MetricData(
            name='ProcessingLatency',
            value=duration_ms,
            unit='Milliseconds',
            dimensions={'UserId': user_id[:8]}  # Anonymize
        ))

    def record_quality_score(self, score: float):
        """Record quality score distribution."""
        self.buffer.append(MetricData(
            name='QualityScore',
            value=score,
            unit='None'
        ))

    def record_acceptance(self, accepted: bool):
        """Record acceptance/rejection."""
        self.buffer.append(MetricData(
            name='MeasurementAccepted' if accepted else 'MeasurementRejected',
            value=1,
            unit='Count'
        ))

    def flush(self):
        """Send buffered metrics to CloudWatch."""
        if not self.buffer:
            return

        metric_data = []
        for metric in self.buffer:
            data = {
                'MetricName': metric.name,
                'Value': metric.value,
                'Unit': metric.unit,
                'Timestamp': datetime.utcnow()
            }
            if metric.dimensions:
                data['Dimensions'] = [
                    {'Name': k, 'Value': v}
                    for k, v in metric.dimensions.items()
                ]
            metric_data.append(data)

        # Send in batches of 20 (CloudWatch limit)
        for i in range(0, len(metric_data), 20):
            batch = metric_data[i:i+20]
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=batch
            )

        self.buffer.clear()
```

## 5. Error Handling & Recovery

### 5.1 Circuit Breaker Pattern
```python
# src/utils/circuit_breaker.py
from typing import Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import threading

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to retry."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >
            timedelta(seconds=self.recovery_timeout)
        )

    def _on_success(self):
        """Reset circuit breaker on success."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Increment failure count and possibly open circuit."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
```

## 6. Data Migration Strategy

### 6.1 Migration Script
```python
# scripts/migrate_to_aws.py
import csv
import boto3
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm

class DataMigrator:
    """
    Migrate existing data to AWS service.
    """

    def __init__(self, api_client, batch_size: int = 100):
        self.api_client = api_client
        self.batch_size = batch_size

    def migrate_user_data(self, csv_file: str, user_ids: List[str] = None):
        """Migrate data from CSV to AWS service."""

        # Group measurements by user
        user_measurements = self._load_csv_data(csv_file, user_ids)

        # Process each user
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._process_user, user_id, measurements): user_id
                for user_id, measurements in user_measurements.items()
            }

            for future in tqdm(concurrent.futures.as_completed(futures),
                              total=len(futures)):
                user_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Failed to migrate user {user_id}: {e}")

        return results

    def _load_csv_data(self, csv_file: str,
                       user_ids: List[str]) -> Dict[str, List[Dict]]:
        """Load and group CSV data by user."""
        user_data = {}

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row['user_id']

                if user_ids and user_id not in user_ids:
                    continue

                if user_id not in user_data:
                    user_data[user_id] = []

                user_data[user_id].append({
                    'uuid': row.get('uuid'),
                    'weight': float(row['weight']),
                    'unit': row['unit'],
                    'effectiveDateTime': row['effectiveDateTime'],
                    'source': row['source']
                })

        return user_data

    def _process_user(self, user_id: str,
                     measurements: List[Dict]) -> Dict[str, Any]:
        """Process single user's data."""

        # Sort by timestamp
        measurements.sort(key=lambda x: x['effectiveDateTime'])

        # Call cleanup endpoint
        response = self.api_client.cleanup(user_id, {
            'measurements': measurements,
            'options': {
                'resetState': True,
                'includeQualityScores': True
            }
        })

        return {
            'userId': user_id,
            'processed': len(measurements),
            'accepted': response['acceptedCount'],
            'rejected': response['rejectedCount']
        }
```

## Next Steps

1. **Review Requirements**: Ensure all features meet Java backend needs
2. **Set Up AWS Environment**: Create AWS accounts, configure IAM roles
3. **Implement Core Service**: Start with basic processing logic
4. **Test Integration**: Develop comprehensive test suite
5. **Performance Testing**: Load test with expected volumes
6. **Documentation**: Create API documentation and runbooks
7. **Deployment**: Progressive rollout with monitoring