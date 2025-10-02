# TypeScript Weight Processor Port - Research Findings

## Overview
This document contains detailed research findings from analyzing the Python weight processor codebase to guide the TypeScript port. All algorithms, data structures, and implementation patterns are documented for accurate 1:1 translation.

## 1. Kalman Filter Implementation

### 1.1 Custom Implementation (no pykalman dependency)
**File**: `weight_values/src/core/processing/kalman_filter.py` (214 lines)

The system uses a **custom Kalman filter implementation** that replaced the pykalman dependency. This is excellent for porting - we don't need to find a TypeScript equivalent library.

**State-Space Model**:
```python
# Process model: x_k = F_k * x_{k-1} + w_k
# Measurement model: z_k = H_k * x_k + v_k

# State vector: x = [weight, velocity]  (2x1)
# Observation: z = [weight]  (1x1)

# Matrices:
F =  [1, time_delta_days]  # 2x2 transition matrix
    [0,       1        ]

H = [1, 0]  # 1x2 observation matrix

Q = [transition_cov_weight,  0                    ]  # 2x2 process noise
    [0,                      transition_cov_trend ]

R = [[observation_covariance]]  # 1x1 measurement noise
```

**Key Methods to Port**:

1. **Predict Step**:
```python
def predict(state_mean, state_covariance):
    # x̂_{k|k-1} = F * x_{k-1|k-1}
    predicted_state_mean = F @ state_mean

    # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    predicted_state_covariance = F @ state_covariance @ F.T + Q

    return predicted_state_mean, predicted_state_covariance
```

2. **Update Step** (Joseph Form for numerical stability):
```python
def update(predicted_state_mean, predicted_state_covariance, observation):
    # Innovation: ỹ_k = z_k - H * x̂_{k|k-1}
    innovation = observation - (H @ predicted_state_mean)

    # Innovation covariance: S_k = H * P_{k|k-1} * H^T + R
    innovation_covariance = H @ predicted_state_covariance @ H.T + R

    # Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
    kalman_gain = predicted_state_covariance @ H.T @ np.linalg.inv(innovation_covariance)

    # Updated state: x̂_{k|k} = x̂_{k|k-1} + K_k * ỹ_k
    filtered_state_mean = predicted_state_mean + kalman_gain @ innovation

    # Updated covariance (Joseph form for stability):
    # P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
    I_KH = I - kalman_gain @ H
    filtered_state_covariance = (I_KH @ predicted_state_covariance @ I_KH.T +
                                 kalman_gain @ R @ kalman_gain.T)

    return filtered_state_mean, filtered_state_covariance
```

**Important Notes**:
- Uses **Joseph form** for covariance update (lines 136-138) - more numerically stable than simplified form
- Matrix inversion only needed for innovation covariance (scalar in our case, so trivial)
- Time-varying transition matrix F based on measurement gaps

### 1.2 Kalman Filter Manager
**File**: `weight_values/src/core/processing/kalman.py` (897 lines)

Manages Kalman filter lifecycle:

1. **Initialization** (`initialize_immediate`):
```python
def initialize_immediate(weight, timestamp, kalman_config, observation_covariance=None):
    initial_variance = 0.364  # from config
    kalman_params = {
        "initial_state_mean": [weight, 0],  # [weight, velocity=0]
        "initial_state_covariance": [[initial_variance, 0], [0, 0.001]],
        "transition_covariance": [[0.018, 0], [0, 0.00012]],
        "observation_covariance": [[5.0]]  # or adaptive
    }
    last_state = np.array([[weight, 0]])
    last_covariance = np.array([[[initial_variance, 0], [0, 0.001]]])
```

2. **State Update**:
- Time delta calculation: `max(0.1, min(30.0, (current - last).total_seconds() / 86400))`
- Builds dynamic transition matrix F with time delta
- Converts Decimal types from DynamoDB to float

3. **Prediction** (`predict_next_state`):
```python
# Used for quality scoring - get prediction WITHOUT updating state
def predict_next_state(state, timestamp):
    F = [[1, time_delta_days], [0, 1]]
    x_pred = F @ x_posterior
    P_pred = F @ P_posterior @ F.T + Q
    predicted_weight = x_pred[0]
    innovation_covariance = P_pred[0, 0] + R
    return predicted_weight, innovation_covariance
```

4. **Adaptive Covariances**:
```python
# After reset, use boosted covariances that decay exponentially
def get_adaptive_covariances(measurements_since_reset, config):
    warmup = 10  # measurements
    if measurements_since_reset < warmup:
        factor = exp(-measurements_since_reset / decay_rate)
        weight_mult = 1 + (boost_factor - 1) * factor
        return base_cov * weight_mult
    return base_cov
```

## 2. Quality Scoring System

### 2.1 Unified Quality Scorer
**File**: `weight_values/src/core/processing/unified_quality_scorer.py` (1051 lines)

**Architecture**:
- 5 components with configurable weights
- Weighted geometric mean or harmonic mean
- Each component returns score in [0, 1]
- Final score compared to threshold (default 0.46)

**Component Weights** (from config):
```python
DEFAULT_WEIGHTS = {
    "kalman_fit": 0.40,          # Primary signal
    "temporal_consistency": 0.30,
    "anomaly_detection": 0.20,
    "source_reliability": 0.05,
    "trend_alignment": 0.05
}
```

**Weighted Geometric Mean**:
```python
def _calculate_weighted_geometric_mean(components):
    product = 1.0
    weight_sum = 0.0
    for name, score in components.items():
        weight = weights[name]
        product *= score ** weight
        weight_sum += weight
    return product ** (1.0 / weight_sum)
```

### 2.2 Component Algorithms

#### 2.2.1 Kalman Fit (40% weight)
```python
def calculate_kalman_fit(weight, kalman_prediction, innovation_covariance, kalman_state):
    innovation = weight - kalman_prediction
    normalized_innovation = abs(innovation) / sqrt(innovation_covariance)

    # Chi-squared test
    chi_squared = normalized_innovation ** 2
    p_value = 1 - chi2_cdf(chi_squared, df=1)  # NEEDS APPROXIMATION IN TS

    # Exponential decay score
    if in_adaptive_period:
        score = exp(-0.2 * normalized_innovation)  # More lenient
    else:
        score = exp(-0.5 * normalized_innovation)  # Standard

    # Time-based decay for gap tolerance
    if days_since_last > 0:
        decay_factor = min(1.0, days_since_last / 30.0)
        score = score + (1.0 - score) * decay_factor  # Blend to 1.0

    return score
```

**Chi-Squared CDF Challenge**: TypeScript needs approximation algorithm
- Input: x (chi-squared value), df=1 (degrees of freedom)
- Output: probability value in [0, 1]
- Options: Abramowitz & Stegun approximation, lookup table, or existing JS library

#### 2.2.2 Temporal Consistency (30% weight)
```python
def calculate_temporal_consistency(weight, previous_weight, time_diff_hours):
    # Exponential growth of acceptable change
    max_acceptable = 0.5 + 4.5 * (1 - exp(-time_diff_hours / 48))
    weight_change = abs(weight - previous_weight)

    if weight_change <= max_acceptable:
        score = 0.8 + 0.2 * exp(-weight_change / max_acceptable)
    else:
        excess_ratio = (weight_change - max_acceptable) / max_acceptable
        score = 0.8 * exp(-excess_ratio)

    # Clamp to [0.2, 1.0]
    return max(0.2, min(1.0, score))
```

#### 2.2.3 Anomaly Detection (20% weight)
Complex multi-stage algorithm with:

1. **Absolute Physiological Bounds**:
```python
ABSOLUTE_MIN_WEIGHT = 30  # kg - hard reject
ABSOLUTE_MAX_WEIGHT = 400  # kg - hard reject
SUSPICIOUS_MIN_WEIGHT = 40  # kg - penalize
SUSPICIOUS_MAX_WEIGHT = 300  # kg - penalize
```

2. **Time-Based Physiological Limits** (piecewise function):
```python
def _calculate_max_physiological_change(time_hours):
    if time_hours < 0.0167:  # < 1 minute
        return 0.5  # kg (scale variance)
    elif time_hours < 0.0833:  # < 5 minutes
        return linear_interpolate(0.5, 1.0, time_hours, 1/60, 5/60)
    elif time_hours < 1:  # < 1 hour
        return 0.5 + 0.5 * log(minutes/5) / log(12)  # logarithmic growth
    elif time_hours <= 6:
        return 1.0 + (3.0 - 1.0) * log(1 + (hours - 1)) / log(6)
    elif time_hours <= 24:
        return 3.0 + (4.0 - 3.0) * log(1 + (hours - 6)/6) / log(4)
    elif time_hours <= 168:  # 7 days
        return min(3.5, 2.0 * sqrt(days))
    else:
        return 3.5 + (days - 7) * 0.5  # sustained rate
```

3. **Duplicate Detection**:
```python
# < 5 seconds with same weight (within 50g) = duplicate
if time_diff_seconds < 5 and weight_change < 0.05:
    return 0.0  # Reject
```

4. **Rapid-Fire Detection**:
```python
# < 5 minutes: adaptive threshold with source factor
max_allowed = 0.5 + 0.5 * (1 - exp(-time_minutes / 2))
if device_source:
    max_allowed *= 1.5  # More lenient
if weight_change > max_allowed * 2:
    return 0.0  # Reject impossible change
```

5. **Burst Pattern Detection**:
```python
# 5+ measurements in 30 minutes
if burst_count >= 5:
    burst_penalty = max(0.6, 1.0 - (burst_count - 4) * 0.1)
    score *= burst_penalty
```

6. **Percentage-Based Changes** (for 3-30 day periods):
```python
if 72 < time_hours <= 720:
    percent_change = (weight_change / previous_weight) * 100
    time_factor = sqrt(time_hours / 720)  # Smoother scaling
    allowed_percent = 15 * time_factor  # Max 15% per month
    if percent_change > allowed_percent:
        # Apply penalties based on severity
```

7. **Sustained Pattern Analysis**:
```python
# Check if changes follow consistent direction (all gains or all losses)
differences = [weights[i+1] - weights[i] for i in range(len-1)]
if all_positive or all_negative:
    consistency_score = 1.0
else:
    cv = std(differences) / mean_abs(differences)  # Coefficient of variation
    consistency_score = exp(-cv * 0.5)
```

**Weighted Average of Penalties**:
```python
# Instead of multiplying all penalties, use weighted average
# This prevents overly harsh rejections
penalty_components = [...]  # Various penalty scores
penalty_weights = [...]     # Importance of each penalty
weighted_score = sum(p * w for p, w in zip(components, weights)) / sum(weights)
score *= max(0.25, weighted_score)  # Floor at 0.25 unless impossible change
```

#### 2.2.4 Source Reliability (5% weight)
```python
def calculate_source_reliability(source):
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    noise_multiplier = profile["noise_multiplier"]  # 0.5 - 3.0
    # Invert and normalize to [0.2, 1.0]
    reliability = 1.0 - ((noise_multiplier - 0.5) / 2.5)
    return max(0.2, min(1.0, reliability))
```

#### 2.2.5 Trend Alignment (5% weight)
```python
def calculate_trend_alignment(weight, kalman_state, recent_weights):
    # Need at least 5 measurements
    if len(recent_weights) < 5:
        return 0.8  # Neutral-high

    # Linear regression on recent weights
    x = range(len(recent_weights))
    slope, intercept = polyfit(x, recent_weights, degree=1)
    predicted_next = slope * len(recent_weights) + intercept

    # Calculate deviation from trend
    deviation = abs(weight - predicted_next)
    std_dev = std(residuals)  # or min 0.5 kg
    normalized_deviation = deviation / std_dev

    # Exponential decay score
    k = 0.3  # Decay constant (configurable)
    score = exp(-k * normalized_deviation)
    return max(0.3, score)
```

**Statistical Functions Needed**:
- `polyfit(x, y, degree=1)`: Linear regression
- Calculate residuals and standard deviation

## 3. Reset Management System

### 3.1 Reset Types (Enum)
```typescript
enum ResetType {
    INITIAL = 'initial',  // First measurement - most aggressive
    HARD = 'hard',         // 30+ day gap - aggressive
    SOFT = 'soft'          // Manual data with significant change - gentle
}
```

### 3.2 Reset Detection Logic
**File**: `weight_values/src/core/processing/kalman.py` (lines 589-655)

```python
def should_trigger_reset(state, weight, timestamp, source, config) -> Optional[ResetType]:
    # Priority order: Initial > Hard > Soft

    # 1. Initial reset (no Kalman params yet)
    if not state or not state.get("kalman_params"):
        return ResetType.INITIAL

    # 2. Hard reset (30+ day gap)
    hard_enabled = config["kalman"]["reset"]["hard"]["enabled"]
    if hard_enabled:
        last_timestamp = state.get("last_accepted_timestamp") or state.get("last_timestamp")
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0
        threshold = config["kalman"]["reset"]["hard"]["gap_threshold_days"]  # 30
        if gap_days >= threshold:
            return ResetType.HARD

    # 3. Soft reset (manual data with significant change)
    soft_enabled = config["kalman"]["reset"]["soft"]["enabled"]
    if soft_enabled:
        manual_sources = {"internal-questionnaire", "initial-questionnaire", ...}
        if source in manual_sources:
            last_weight = state.get("last_raw_weight")
            weight_change = abs(weight - last_weight)
            min_change = config["kalman"]["reset"]["soft"]["min_weight_change_kg"]  # 5
            if weight_change >= min_change:
                # Check cooldown (prevent rapid resets)
                cooldown_days = config["kalman"]["reset"]["soft"]["cooldown_days"]  # 3
                last_reset_time = get_last_reset_timestamp(state)
                if not last_reset_time or gap_since_reset > cooldown_days:
                    return ResetType.SOFT

    return None
```

### 3.3 Reset Parameters
**File**: `weight_values/src/core/processing/kalman.py` (lines 677-753)

Each reset type has different adaptation parameters:

```python
RESET_PARAMETERS = {
    ResetType.INITIAL: {
        # Multipliers for Kalman parameters
        "initial_variance_multiplier": 10,
        "weight_noise_multiplier": 50,
        "trend_noise_multiplier": 50,
        "observation_noise_multiplier": 20,
        # Adaptation duration
        "adaptation_measurements": 10,
        "adaptation_days": 10,
        "adaptation_decay_rate": 2.5,
        # Quality scoring adjustments
        "quality_acceptance_threshold": 0.25,
        "quality_safety_weight": 0.50,
        "quality_plausibility_weight": 0.05,
        "quality_consistency_weight": 0.05,
        "quality_reliability_weight": 0.40
    },
    ResetType.HARD: {
        "initial_variance_multiplier": 3,
        "weight_noise_multiplier": 5,
        "trend_noise_multiplier": 50,
        "observation_noise_multiplier": 0.7,
        "adaptation_measurements": 10,
        "adaptation_days": 7,
        "adaptation_decay_rate": 2.5,
        "quality_acceptance_threshold": 0.35,
        ...
    },
    ResetType.SOFT: {
        "initial_variance_multiplier": 2,
        "weight_noise_multiplier": 5,
        "trend_noise_multiplier": 20,
        "observation_noise_multiplier": 0.7,
        "adaptation_measurements": 15,
        "adaptation_days": 10,
        ...
    }
}
```

### 3.4 Reset Execution
```python
def perform_reset(state, reset_type, timestamp, weight, source, config):
    reset_params = get_reset_parameters(reset_type, config)

    # Calculate gap if applicable
    gap_days = None
    if last_timestamp:
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400.0

    # Create reset event
    reset_event = {
        "timestamp": timestamp,
        "type": reset_type.value,
        "source": source,
        "weight": weight,
        "last_weight": state.get("last_raw_weight"),
        "gap_days": gap_days,
        "reason": get_reset_reason(reset_type, gap_days, weight, state),
        "parameters": reset_params
    }

    # Create new state with reset
    new_state = {
        "kalman_params": None,  # Will be re-initialized
        "last_state": None,
        "last_covariance": None,
        "measurements_since_reset": 0,
        "reset_type": reset_type.value,
        "reset_parameters": reset_params,
        "reset_timestamp": timestamp,
        "reset_events": state.get("reset_events", []) + [reset_event],
        "last_timestamp": state.get("last_timestamp"),  # Preserve
        "last_source": state.get("last_source"),
        "last_raw_weight": state.get("last_raw_weight"),
        "last_accepted_timestamp": state.get("last_accepted_timestamp"),
        "measurement_history": []  # Clear history
    }

    return new_state, reset_event
```

### 3.5 Adaptive Period Calculation
```python
def is_in_adaptive_period(state, timestamp):
    reset_timestamp = state.get("reset_timestamp")
    if not reset_timestamp:
        return False, None

    reset_params = state.get("reset_parameters", {})

    # Check measurements-based
    measurements_since = state.get("measurements_since_reset", 0)
    adaptation_measurements = reset_params.get("adaptation_measurements", 10)

    # Check time-based
    days_since = (timestamp - reset_timestamp).total_seconds() / 86400.0
    adaptation_days = reset_params.get("adaptation_days", 7)

    # In adaptive period if EITHER condition is met
    if measurements_since < adaptation_measurements or days_since < adaptation_days:
        return True, reset_params

    return False, None
```

```python
def get_adaptive_factor(state, timestamp):
    # Returns 0.0 (fully adaptive) to 1.0 (normal operation)
    measurements_since = state.get("measurements_since_reset", 0)
    decay_rate = reset_params.get("adaptation_decay_rate", 3)
    factor = 1.0 - exp(-measurements_since / decay_rate)
    return min(1.0, max(0.0, factor))
```

## 4. Replay System

### 4.1 ReplayBuffer
**File**: `weight_values/src/core/replay/replay_buffer.py`

Thread-safe buffer for 24-hour measurement windows:

```python
class ReplayBuffer:
    def __init__(self, config):
        self.buffer_hours = config.get("buffer_hours", 24)
        self.max_buffer_measurements = config.get("max_buffer_measurements", 100)
        self.buffers = {}  # user_id -> buffer_data
        self._lock = threading.RLock()

    def add_measurement(self, user_id, measurement):
        with self._lock:
            buffer_data = self.buffers[user_id]
            buffer_data["measurements"].append(measurement)
            # Update timestamps, enforce limits, check triggers
            return {
                "success": True,
                "buffer_ready": should_trigger,
                "buffer_size": len(measurements)
            }
```

**TypeScript Note**: Replace threading.RLock with standard locking or async patterns (not needed for single-threaded CLI)

### 4.2 ReplayManager
**File**: `weight_values/src/core/replay/replay_manager.py` (636 lines)

Manages state restoration and chronological replay:

**Main Workflow**:
```python
def replay_clean_measurements(user_id, clean_measurements, buffer_start_time):
    # Step 0: Check for concurrent replay (prevent race condition)
    if state.get("replay_in_progress"):
        return {"success": False, "error": "Replay already in progress"}

    # Step 1: Create backup of current state (for rollback)
    backup_state = copy.deepcopy(current_state)

    # Step 1.5: Set replay_in_progress flag
    state["replay_in_progress"] = True
    state["replay_started_at"] = datetime.now().isoformat()

    try:
        # Step 2: Restore state to before buffer start
        snapshot = db.get_snapshot(user_id, buffer_start_time)
        if not snapshot:
            return {"success": False, "error": "No snapshot found"}

        # Validate snapshot
        if not _validate_snapshot(snapshot):
            return {"success": False, "error": "Invalid snapshot"}

        # Restore atomically
        db.save_state(user_id, snapshot)

        # Step 2.5: Trajectory continuity check
        backup_weight = get_weight_from_state(backup_state)
        restored_weight = get_weight_from_state(snapshot)
        if abs(backup_weight - restored_weight) > 15.0:
            # Rollback - would cause impossible jump
            db.save_state(user_id, backup_state)
            return {"success": False, "error": "Trajectory jump exceeds 15kg"}

        # Step 3: Replay measurements chronologically
        sorted_measurements = sorted(clean_measurements, key=lambda m: m["timestamp"])
        for measurement in sorted_measurements:
            # Process through normal pipeline
            result = process_measurement(user_id, weight, timestamp, source, config, db=db)
            # Continue even if rejected (normal behavior)

        # Step 4: Verify state saved
        final_state = db.get_state(user_id)
        if not final_state:
            db.save_state(user_id, backup_state)  # Rollback
            return {"success": False, "error": "State verification failed"}

        # Step 5: Clear replay flag and backup
        state["replay_in_progress"] = False
        del backup_state

        return {
            "success": True,
            "measurements_replayed": len(sorted_measurements),
            "final_state": final_state
        }

    except Exception as e:
        # Emergency rollback
        db.save_state(user_id, backup_state)
        state["replay_in_progress"] = False
        return {"success": False, "error": str(e)}
```

**Retry Logic**:
```python
def _restore_state_to_buffer_start(user_id, buffer_start_time):
    max_retries = 3
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        result = db.check_and_restore_snapshot(user_id, buffer_start_time)
        if result["success"]:
            # Validate snapshot before using
            if _validate_snapshot(result["snapshot"]):
                return result

        # Don't retry if snapshot doesn't exist (not transient)
        if "No snapshot found" in result.get("error", ""):
            break

    return {"success": False, "error": "All retries failed"}
```

### 4.3 Outlier Detection
**File**: `weight_values/src/core/processing/outlier_detection.py`

Multiple statistical methods to identify outliers before replay:

```python
class OutlierDetector:
    def detect_outliers(measurements, user_id):
        # 1. Protect high-quality measurements
        protected_indices = set()
        for i, m in enumerate(measurements):
            if m.get("quality_score", 0) > 0.7:
                protected_indices.add(i)

        # 2. Statistical methods
        iqr_outliers = _detect_iqr_outliers(weights)
        zscore_outliers = _detect_zscore_outliers(weights)
        temporal_outliers = _detect_temporal_outliers(measurements)
        kalman_outliers = _detect_kalman_outliers(measurements, user_id)

        # 3. AND logic: outlier if NOT protected AND fails multiple tests
        final_outliers = set()
        for i in range(len(measurements)):
            if i in protected_indices:
                continue
            if i not in (iqr_outliers & zscore_outliers):
                continue
            if kalman_outliers and i not in kalman_outliers:
                continue
            final_outliers.add(i)

        return final_outliers
```

**IQR Method**:
```python
def _detect_iqr_outliers(weights):
    q1 = percentile(weights, 25)
    q3 = percentile(weights, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {i for i, w in enumerate(weights) if w < lower or w > upper}
```

**Modified Z-Score (MAD-based)**:
```python
def _detect_zscore_outliers(weights):
    median = np.median(weights)
    mad = np.median(np.abs(weights - median))
    if mad == 0:
        return set()
    # Modified z-scores
    z_scores = 0.6745 * (weights - median) / mad
    return {i for i, z in enumerate(z_scores) if abs(z) > 3.0}
```

**Temporal Consistency**:
```python
def _detect_temporal_outliers(sorted_measurements):
    outliers = set()
    for i in range(1, len(measurements)):
        prev = measurements[i-1]
        curr = measurements[i]
        time_diff_hours = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 3600
        weight_change = abs(curr["weight"] - prev["weight"])
        max_change = calculate_max_physiological_change(time_diff_hours)
        percent_change = (weight_change / prev["weight"]) * 100
        max_percent = 30 * (time_diff_hours / 720)  # 30% per month
        if weight_change > max_change or percent_change > max_percent:
            outliers.add(i)
    return outliers
```

## 5. State Management

### 5.1 In-Memory Database
**File**: `weight_values/src/core/database/database.py` (250 lines)

```typescript
class ProcessorStateDB {
    states: Map<string, ProcessorState> = new Map();
    snapshots: Map<string, Snapshot[]> = new Map();

    getState(userId: string): ProcessorState | null {
        return this.states.get(userId) ? deepCopy(this.states.get(userId)) : null;
    }

    saveState(userId: string, state: ProcessorState): void {
        this.states.set(userId, deepCopy(state));
    }

    deleteState(userId: string): boolean {
        return this.states.delete(userId) && this.snapshots.delete(userId);
    }

    createInitialState(): ProcessorState {
        return {
            kalmanParams: null,
            lastState: null,
            lastCovariance: null,
            lastTimestamp: null,
            lastAcceptedTimestamp: null,
            lastSource: null,
            lastRawWeight: null,
            measurementHistory: [],
            resetEvents: [],
            measurementsSinceReset: 0
        };
    }

    saveStateSnapshot(userId: string, timestamp: Date): boolean {
        const state = this.states.get(userId);
        if (!state) return false;

        if (!this.snapshots.has(userId)) {
            this.snapshots.set(userId, []);
        }

        const snapshot = {
            timestamp,
            snapshotTime: timestamp.toISOString(),
            state: deepCopy(state)
        };

        const userSnapshots = this.snapshots.get(userId)!;
        userSnapshots.push(snapshot);
        userSnapshots.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
        this.snapshots.set(userId, userSnapshots.slice(-10)); // Keep last 10

        return true;
    }

    getSnapshot(userId: string, timestamp: Date): ProcessorState | null {
        const userSnapshots = this.snapshots.get(userId);
        if (!userSnapshots || userSnapshots.length === 0) return null;

        const suitable = userSnapshots.filter(s => s.timestamp < timestamp);
        if (suitable.length === 0) return null;

        const latest = suitable[suitable.length - 1];
        return deepCopy(latest.state);
    }

    getMeasurementsInWindow(userId: string, start: Date, end: Date): Measurement[] {
        const state = this.getState(userId);
        if (!state || !state.measurementHistory) return [];

        return state.measurementHistory
            .filter(m => {
                const ts = typeof m.timestamp === 'string'
                    ? new Date(m.timestamp)
                    : m.timestamp;
                return ts >= start && ts < end;
            })
            .map(m => ({
                timestamp: typeof m.timestamp === 'string' ? new Date(m.timestamp) : m.timestamp,
                weight: m.weight,
                source: m.source || 'unknown',
                unit: m.unit || 'kg',
                metadata: m.metadata || {}
            }));
    }

    checkAndRestoreSnapshot(userId: string, bufferStartTime: Date): {success: boolean, snapshot?: ProcessorState, error?: string} {
        const snapshot = this.getSnapshot(userId, bufferStartTime);
        if (snapshot) {
            this.states.set(userId, deepCopy(snapshot));
            return {
                success: true,
                snapshot,
                snapshotTimestamp: snapshot.lastTimestamp || bufferStartTime
            };
        }
        return {
            success: false,
            error: `No snapshot found before ${bufferStartTime.toISOString()}`
        };
    }
}
```

### 5.2 State Structure
```typescript
interface ProcessorState {
    // Kalman filter state
    kalmanParams: KalmanParams | null;
    lastState: number[] | null;  // [weight, velocity]
    lastCovariance: number[][] | null;  // 2x2 matrix
    lastTimestamp: Date | null;
    lastRawWeight: number | null;
    lastSource: string | null;
    lastAcceptedTimestamp: Date | null;

    // Reset tracking
    measurementsSinceReset: number;
    resetType: 'initial' | 'hard' | 'soft' | null;
    resetParameters: ResetParameters | null;
    resetTimestamp: Date | null;
    resetEvents: ResetEvent[];

    // History
    measurementHistory: MeasurementHistoryEntry[];

    // Replay tracking (transient)
    replayInProgress?: boolean;
    replayStartedAt?: string;

    // Temporal baseline (for quality scoring)
    temporalBaseline?: {
        lastWeight: number;
        lastTimestamp: string;
        rollingAvgChangeRate?: number;
    };
}
```

## 6. Main Processing Pipeline

### 6.1 Processor Orchestration
**File**: `weight_values/src/core/processing/processor.py` (720 lines)

Single function that coordinates all processing:

```python
def process_measurement(user_id, weight, timestamp, source, config, unit='kg', db=None, user_height_m=None):
    # Step 1: Data cleaning (unit conversion, validation)
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit, user_height_m
    )
    if cleaned_weight is None:
        return {"accepted": False, "reason": "preprocessing_failed"}

    # Step 2: Load or create user state
    state = db.get_state(user_id) or db.create_initial_state()

    # Step 3: Check for reset
    reset_type = ResetManager.should_trigger_reset(state, cleaned_weight, timestamp, source, config)
    if reset_type:
        state, reset_event, reset_occurred = _handle_reset_with_transaction(
            user_id, state, reset_type, timestamp, cleaned_weight, source, config
        )

    # Step 4: Initialize Kalman if needed
    if not state.get("kalman_params"):
        # Get adaptive parameters for post-reset period
        adaptive_config = get_adaptive_kalman_params(reset_timestamp, timestamp, kalman_config, state=state)

        # Initialize with first measurement
        kalman_state = KalmanFilterManager.initialize_immediate(
            cleaned_weight, timestamp, adaptive_config, observation_covariance
        )
        state.update(kalman_state)

        result = KalmanFilterManager.create_result(state, cleaned_weight, timestamp, source, True, obs_cov)
        kalman_already_updated = True

    # Step 5: Quality scoring
    quality_scorer = UnifiedQualityScorer(config=quality_config)

    # Get Kalman prediction (predict WITHOUT update)
    kalman_prediction, innovation_covariance = KalmanFilterManager.predict_next_state(state, timestamp)

    # Apply source-specific noise multiplier
    noise_multiplier = get_noise_multiplier(source)
    if innovation_covariance and noise_multiplier != 1.0:
        base_obs_cov = state["kalman_params"]["observation_covariance"][0][0]
        predicted_cov_00 = innovation_covariance - base_obs_cov
        innovation_covariance = predicted_cov_00 + (base_obs_cov * noise_multiplier)

    # Calculate quality score
    quality_score = quality_scorer.calculate_quality_score(
        weight=cleaned_weight,
        source=source,
        kalman_state=state,
        kalman_prediction=kalman_prediction,
        innovation_covariance=innovation_covariance,
        previous_weight=previous_weight,
        time_diff_hours=time_diff_hours,
        recent_weights=recent_weights,
        recent_timestamps=recent_timestamps,
        user_height_m=user_height
    )

    if not quality_score.accepted:
        return {
            "accepted": False,
            "reason": quality_score.rejection_reason,
            "quality_score": quality_score.overall,
            "quality_components": quality_score.components
        }

    # Step 6: Kalman update (if not already done during initialization)
    if not kalman_already_updated:
        adaptive_config = get_adaptive_kalman_params(reset_timestamp, timestamp, kalman_config, state=state)
        observation_covariance = adaptive_config["observation_covariance"] * noise_multiplier

        # Apply trend limiting (clamp to ±5kg/week = ±0.714kg/day)
        current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
        if current_trend and abs(current_trend) > 0.714:
            # Clamp trend before update
            limited_trend = 0.714 if current_trend > 0 else -0.714
            state["last_state"][-1][1] = limited_trend

        state = KalmanFilterManager.update_state(state, cleaned_weight, timestamp, source, {}, obs_cov)

        # Apply trend limiting after update
        current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
        if current_trend and abs(current_trend) > 0.714:
            limited_trend = 0.714 if current_trend > 0 else -0.714
            state["last_state"][-1][1] = limited_trend

        result = KalmanFilterManager.create_result(state, cleaned_weight, timestamp, source, True, obs_cov)

    # Step 7: Add metadata
    result.update({
        "quality_score": quality_score.overall,
        "quality_components": quality_score.components,
        "preprocessing": preprocess_metadata,
        "noise_multiplier": noise_multiplier,
        "stage": "accepted"
    })

    # Step 8: Update measurement history
    state["measurement_history"].append({
        "weight": cleaned_weight,
        "timestamp": timestamp.isoformat(),
        "quality_score": quality_score.overall,
        "source": source
    })
    state["measurement_history"] = state["measurement_history"][-30:]  # Keep last 30

    # Step 9: Save state
    state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1
    state["last_source"] = source
    state["last_timestamp"] = timestamp
    state["last_accepted_timestamp"] = timestamp
    state["last_raw_weight"] = cleaned_weight

    db.save_state(user_id, state)

    # Step 10: Create periodic snapshot (if interval elapsed)
    _maybe_create_periodic_snapshot(db, user_id, timestamp, config)

    return result
```

## 7. Critical Algorithms for TypeScript

### 7.1 Matrix Operations

**2x2 Matrix Inversion** (analytical formula):
```typescript
function invert2x2(matrix: number[][]): number[][] {
    const [[a, b], [c, d]] = matrix;
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-10) {
        throw new Error("Matrix is singular");
    }
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ];
}
```

**Matrix Multiplication**:
```typescript
function matrixMultiply(a: number[][], b: number[][]): number[][] {
    const rowsA = a.length, colsA = a[0].length;
    const rowsB = b.length, colsB = b[0].length;
    if (colsA !== rowsB) {
        throw new Error("Matrix dimensions incompatible");
    }
    const result: number[][] = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));
    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}
```

**Matrix Transpose**:
```typescript
function transpose(matrix: number[][]): number[][] {
    const rows = matrix.length, cols = matrix[0].length;
    const result: number[][] = Array(cols).fill(0).map(() => Array(rows).fill(0));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
```

### 7.2 Statistical Functions

**Chi-Squared CDF Approximation** (Abramowitz & Stegun):
```typescript
// For df=1 (our case), chi-squared CDF can be computed using normal CDF
function chi2Cdf(x: number, df: number): number {
    if (df !== 1) {
        throw new Error("Only df=1 supported (can extend if needed)");
    }
    // For df=1: chi2(x) is distributed as the square of a standard normal
    // P(Chi2 <= x) = 2 * P(Z <= sqrt(x)) - 1
    // where Z ~ N(0,1)
    if (x <= 0) return 0;
    const z = Math.sqrt(x);
    return 2 * normalCdf(z) - 1;
}

// Normal CDF using error function approximation
function normalCdf(x: number): number {
    // Using error function: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1 + erf(x / Math.SQRT2));
}

// Error function approximation (Abramowitz & Stegun 7.1.26)
function erf(x: number): number {
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
}
```

**Linear Regression (Polyfit degree=1)**:
```typescript
function linearRegression(x: number[], y: number[]): {slope: number, intercept: number} {
    const n = x.length;
    if (n !== y.length || n === 0) {
        throw new Error("Invalid input");
    }

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {slope, intercept};
}
```

**Percentile**:
```typescript
function percentile(values: number[], p: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}
```

**Median**:
```typescript
function median(values: number[]): number {
    return percentile(values, 50);
}
```

**Variance & Standard Deviation**:
```typescript
function variance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => (v - mean) ** 2);
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
}

function std(values: number[]): number {
    return Math.sqrt(variance(values));
}
```

### 7.3 Utility Functions

**Deep Copy**:
```typescript
function deepCopy<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
}
```

**Date Parsing**:
```typescript
function parseTimestamp(dateStr: string): Date {
    if (!dateStr) {
        return new Date();
    }

    // Handle ISO format with Z or timezone
    if (dateStr.includes('T')) {
        return new Date(dateStr.replace('Z', '+00:00'));
    }

    // Handle space-separated format
    if (dateStr.includes(' ')) {
        // Try with milliseconds, then without
        const formats = [
            /(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)/,
            /(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/
        ];
        for (const fmt of formats) {
            const match = dateStr.match(fmt);
            if (match) {
                return new Date(dateStr);
            }
        }
    }

    // Date only
    return new Date(dateStr);
}
```

## 8. Configuration Management

### 8.1 TOML Parsing
**Dependencies**: `@iarna/toml` (well-established library)

```typescript
import TOML from '@iarna/toml';
import { readFileSync } from 'fs';

class ConfigManager {
    static loadConfig(source: 'file' | 'env' = 'file', configPath?: string): Config {
        if (source === 'file') {
            const path = configPath || 'config.toml';
            const content = readFileSync(path, 'utf-8');
            const parsed = TOML.parse(content) as any;

            // Transform to expected structure
            return {
                kalman: parsed.kalman || {},
                quality_scoring: parsed.quality_scoring || {},
                sources: parsed.sources || {},
                processing: parsed.processing || {},
                replay: parsed.replay || {},
                snapshot: parsed.snapshot || {},
                database: parsed.database || {}
            };
        }

        // TODO: env source if needed
        throw new Error("Env source not yet implemented");
    }
}
```

### 8.2 Constants
**File**: `weight_values/src/core/constants.py`

Port all constants to TypeScript:

```typescript
// constants.ts

export const PHYSIOLOGICAL_LIMITS = {
    ABSOLUTE_MIN_WEIGHT: 30.0,
    ABSOLUTE_MAX_WEIGHT: 400.0,
    SUSPICIOUS_MIN_WEIGHT: 40.0,
    SUSPICIOUS_MAX_WEIGHT: 300.0,
    DEFAULT_HEIGHT_M: 1.67,
    MAX_DAILY_CHANGE_KG: 2.0,
    MAX_WEEKLY_CHANGE_KG: 3.5,
    TYPICAL_DAILY_VARIATION_KG: 1.5,
    MAX_SUSTAINED_DAILY_KG: 0.5,
    MAX_CHANGE_1H: 1.0,
    MAX_CHANGE_6H: 3.0,
    MAX_CHANGE_24H: 4.0,
    MAX_CHANGE_1MIN: 0.5,
    MAX_CHANGE_5MIN: 1.0,
    MAX_MONTHLY_PERCENT: 15,
    LIMIT_TOLERANCE: 0.1,
    SUSTAINED_TOLERANCE: 0.25,
    SESSION_VARIANCE: 2
};

export const SUPPORTED_WEIGHT_UNITS = new Set([
    'kg', 'kilogram', 'kilograms',
    'g', 'gram', 'grams',
    'lb', 'lbs', 'pound', 'pounds',
    'st', 'stone', 'stones'
]);

export const BMI_LIMITS = {
    CRITICAL_LOW: 15.0,
    SEVERE_LOW: 16.0,
    UNDERWEIGHT: 18.5,
    OVERWEIGHT: 25.0,
    OBESE: 30.0,
    SEVERE_OBESE: 35.0,
    MORBID_OBESE: 40.0,
    CRITICAL_HIGH: 50.0
};

export const KALMAN_DEFAULTS = {
    initial_variance: 0.364,
    transition_covariance_weight: 0.018,
    transition_covariance_trend: 0.00015,
    observation_covariance: 3.4
};

export const QUESTIONNAIRE_SOURCES = new Set([
    'internal-questionnaire',
    'initial-questionnaire',
    'care-team-upload',
    'questionnaire'
]);
```

**Note**: SOURCE_PROFILES will be loaded from config.toml, not hardcoded

## 9. CSV Processing

### 9.1 CSV Reading
**Dependencies**: `csv-parse` (mature library)

```typescript
import { parse } from 'csv-parse/sync';
import { readFileSync } from 'fs';

interface LoadOptions {
    maxUsers: number;
    maxRows: number;
    minReadings: number;
}

interface CsvRow {
    id?: string;
    measurement_id?: string;
    user_id: string;
    value_quantity?: string;
    weight?: string;
    unit: string;
    effective_date_time?: string;
    effectiveDateTime?: string;
    source_type: string;
    [key: string]: any;
}

function loadCsvData(csvPath: string, options: LoadOptions): {
    userMeasurements: Map<string, Measurement[]>;
    originalRows: CsvRow[];
} {
    const content = readFileSync(csvPath, 'utf-8');
    const records = parse(content, {
        columns: true,
        skip_empty_lines: true,
        trim: true
    }) as CsvRow[];

    const userMeasurements = new Map<string, Measurement[]>();
    const originalRows: CsvRow[] = [];
    const stats = {
        total_rows: 0,
        invalid_weight: 0,
        parse_errors: 0,
        unit_rejected: 0,
        rejected_units: new Map<string, number>(),
        bsa_measurements: 0,
        missing_data: 0
    };

    for (const [rowIndex, row] of records.entries()) {
        stats.total_rows++;

        if (options.maxRows > 0 && stats.total_rows > options.maxRows) {
            break;
        }

        // Extract fields (handle both old and new column names)
        const measurementId = row.id || row.measurement_id;
        const userId = row.user_id;
        if (!userId || !measurementId) {
            stats.missing_data++;
            continue;
        }

        // Parse weight
        const weightStr = (row.value_quantity || row.weight || '').trim();
        if (!weightStr || weightStr.toUpperCase() === 'NULL') {
            stats.missing_data++;
            continue;
        }

        let weight: number;
        try {
            weight = parseFloat(weightStr);
            if (weight <= 0 || weight > 1000 || isNaN(weight) || !isFinite(weight)) {
                stats.invalid_weight++;
                continue;
            }
        } catch {
            stats.parse_errors++;
            continue;
        }

        // Other fields
        const dateStr = row.effective_date_time || row.effectiveDateTime || '';
        const source = row.source_type || 'unknown';
        const unit = (row.unit || '').trim();

        // Skip BSA measurements
        if (source.toUpperCase().includes('BSA') || unit === 'm2' || unit === 'm²') {
            stats.bsa_measurements++;
            continue;
        }

        // Validate unit
        if (!unit || !SUPPORTED_WEIGHT_UNITS.has(unit.toLowerCase())) {
            stats.unit_rejected++;
            stats.rejected_units.set(unit || '<missing>',
                (stats.rejected_units.get(unit || '<missing>') || 0) + 1);
            continue;
        }

        // Parse timestamp
        const timestamp = parseTimestamp(dateStr);

        // Create measurement
        const measurement: Measurement = {
            uuid: measurementId,
            weight,
            unit,
            effectiveDateTime: timestamp,
            source,
            metadata: {
                original_row_index: rowIndex,
                csv_row: row
            }
        };

        if (!userMeasurements.has(userId)) {
            userMeasurements.set(userId, []);
        }
        userMeasurements.get(userId)!.push(measurement);

        // Track original row
        originalRows.push({...row, _row_index: rowIndex, _accepted: false});
    }

    // Apply filters (min_readings, max_users)
    // ... (same logic as Python)

    return {userMeasurements, originalRows};
}
```

### 9.2 CSV Writing
```typescript
import { stringify } from 'csv-stringify/sync';
import { writeFileSync } from 'fs';

function writeFilteredCsv(
    originalRows: CsvRow[],
    acceptanceTracker: AcceptanceTracker,
    outputPath: string
): number {
    const fieldnames = Object.keys(originalRows[0] || {}).filter(k => !k.startsWith('_'));

    const acceptedRows = originalRows.filter(row => {
        const userId = row.user_id;
        const timestamp = parseTimestamp(row.effective_date_time || row.effectiveDateTime || '');
        return acceptanceTracker.isAccepted(userId, timestamp.toISOString());
    });

    const csvContent = stringify(acceptedRows, {
        header: true,
        columns: fieldnames
    });

    writeFileSync(outputPath, csvContent, 'utf-8');

    return acceptedRows.length;
}
```

## 10. Testing Strategy

### 10.1 Validation Tests
Create test suite that compares TypeScript output to Python output:

```typescript
// test/validation.test.ts
import { describe, test, expect } from 'bun:test';
import { loadCsvData } from '../src/local_main';
import { WeightProcessorService } from '../src/services/weight_processor_service';
import { readFileSync } from 'fs';

describe('Python vs TypeScript Validation', () => {
    test('same CSV produces same results', () => {
        // Load CSV
        const {userMeasurements} = loadCsvData('test/fixtures/sample.csv', {
            maxUsers: 10,
            maxRows: 1000,
            minReadings: 20
        });

        // Process with TS
        const service = new WeightProcessorService();
        const tsResults = new Map();
        for (const [userId, measurements] of userMeasurements) {
            const result = service.processBatch(userId, measurements);
            tsResults.set(userId, result);
        }

        // Load Python results
        const pythonResults = JSON.parse(readFileSync('test/fixtures/python_output.json', 'utf-8'));

        // Compare
        for (const [userId, tsResult] of tsResults) {
            const pyResult = pythonResults[userId];

            expect(tsResult.measurementsProcessed).toBe(pyResult.measurements_processed);
            expect(tsResult.measurementsAccepted).toBe(pyResult.measurements_accepted);

            // Compare each measurement result
            for (let i = 0; i < tsResult.results.length; i++) {
                const tsRes = tsResult.results[i];
                const pyRes = pyResult.results[i];

                expect(tsRes.accepted).toBe(pyRes.accepted);

                if (tsRes.accepted) {
                    // Numerical tolerance: within 0.1%
                    expect(Math.abs(tsRes.kalmanEstimate - pyRes.kalman_estimate)).toBeLessThan(0.001);
                    expect(Math.abs(tsRes.qualityScore - pyRes.quality_score)).toBeLessThan(0.001);
                }
            }
        }
    });
});
```

## 11. Key Implementation Challenges

### 11.1 Numerical Stability
- **Covariance Symmetry**: Ensure covariance matrices stay symmetric (use Joseph form)
- **Positive Definiteness**: Monitor for numerical drift
- **Floating-Point Precision**: JavaScript has 64-bit floats (same as Python), should be fine

### 11.2 Date/Time Handling
- Python uses timezone-aware `datetime` objects
- TypeScript should use `Date` with UTC
- Be careful with ISO string parsing and timezones

### 11.3 Type Safety
- Avoid `any` types
- Use strict TypeScript mode
- Define comprehensive interfaces for all data structures

### 11.4 Deep Copying
- Python uses `copy.deepcopy()`
- TypeScript needs custom implementation or `JSON.parse(JSON.stringify())` for simple objects
- Watch for Date objects in deep copy (they serialize to strings)

## 12. Dependencies Summary

### Runtime Dependencies
1. **@iarna/toml** - TOML parsing (config.toml)
2. **csv-parse** - CSV reading
3. **csv-stringify** - CSV writing (optional, can use built-in)

### Development Dependencies
1. **typescript** - TypeScript compiler
2. **@types/bun** - Bun type definitions
3. **bun:test** - Testing framework (built into Bun)

### No External Dependencies Needed For
- Matrix operations (implement from scratch)
- Statistical functions (implement from scratch)
- Kalman filter (implement from scratch)
- Date parsing (use built-in Date)

## 13. Performance Considerations

### 13.1 Avoid Premature Optimization
- Focus on correctness first
- Port algorithms 1:1 before optimizing
- Bun is fast - likely faster than Python without optimization

### 13.2 Potential Optimizations (post-validation)
- Use typed arrays for large matrix operations?
- Pre-allocate arrays where size is known
- Memoize expensive calculations
- Batch database operations

## 14. Next Steps

1. **Discussion Phase**: Create solution options for architecture and module organization
2. **Recommendation**: Choose best approach for TypeScript structure
3. **Planning**: Break down into phased implementation tasks
4. **Implementation**: Port modules incrementally with tests

---

## Appendix: File Count and Complexity

**Total Files to Port**: ~30 TypeScript files
- Core processing: 14 files
- Database: 3 files
- Replay: 6 files
- Configuration: 4 files
- Services: 1 file
- Models: 1 file
- CLI: 1 file

**Estimated Lines of Code**: ~5,000-6,000 lines (similar to Python)

**Complexity Distribution**:
- High: Kalman filter, Quality scorer, Replay manager
- Medium: Processor, Reset manager, Outlier detection
- Low: Database, Config, Utils, Constants
