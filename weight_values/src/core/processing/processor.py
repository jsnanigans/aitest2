"""
Simplified weight processor with flattened pipeline.
Single processing function with clear flow.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..database import get_state_db
from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .kalman import (
    KalmanFilterManager,
    ResetManager,
    ResetType,
    get_adaptive_kalman_params,
    get_reset_timestamp,
)
from .persistence_validator import PersistenceValidator
from .reset_transaction import ResetOperation, ResetTransaction
from .validation import DataQualityPreprocessor

logger = logging.getLogger(__name__)
from ..constants import KALMAN_DEFAULTS, PHYSIOLOGICAL_LIMITS, get_noise_multiplier

from .unified_quality_scorer import (
    QualityScore,
    UnifiedQualityScorer,
    MeasurementHistory,
)

# Import type conversion utilities
try:
    from ..utils.type_conversion import ensure_float, ensure_numeric_types
except ImportError:
    # Fallback if not available
    def ensure_float(value):
        """Convert value to float, handling Decimal types."""
        if hasattr(value, "is_finite"):  # Decimal
            return float(value)
        return float(value) if value is not None else 0.0

    def ensure_numeric_types(data):
        """Ensure all numeric values in data are proper Python types."""
        return data


# Verbose logging helpers
import os
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

def _log(message: str):
    """Log processing step if verbose logging enabled."""
    if VERBOSE_LOGGING:
        print(f"[PY] {message}", flush=True)

def _format_num(value: float | None) -> str:
    """Format number to 6 decimal places."""
    if value is None:
        return "None"
    return f"{float(value):.6f}"

def _format_vec(vec) -> str:
    """Format state vector/array."""
    if vec is None:
        return "None"
    if hasattr(vec, 'flatten'):
        flat = vec.flatten()
    elif isinstance(vec, list):
        flat = vec if not isinstance(vec[0], list) else [item for sublist in vec for item in sublist]
    else:
        flat = vec
    return f"[{', '.join(_format_num(float(v)) for v in flat)}]"


def _maybe_create_periodic_snapshot(
    db,
    user_id: str,
    timestamp: datetime,
    config: Dict[str, Any]
) -> bool:
    """
    Create periodic snapshot if interval has elapsed since last snapshot.

    This function checks if enough time has passed since the last snapshot
    and creates a new one if needed. This ensures replay functionality has
    adequate state history even for users without recent resets.

    Args:
        db: State store instance
        user_id: User identifier
        timestamp: Current measurement timestamp
        config: Configuration dictionary

    Returns:
        True if snapshot was created, False otherwise
    """
    try:
        # Get snapshot configuration
        snapshot_config = config.get("snapshot", {})
        snapshot_interval_hours = snapshot_config.get("interval_hours", 24)
        periodic_enabled = snapshot_config.get("periodic_enabled", True)

        if not periodic_enabled:
            return False

        # Get the latest snapshot for this user
        latest_snapshot = db.get_latest_snapshot(user_id)

        # Create snapshot if none exists (initial snapshot)
        if not latest_snapshot:
            db.save_state_snapshot(user_id, timestamp)
            logger.debug(f"Created initial periodic snapshot for user {user_id}")
            return True

        # Check time since last snapshot
        last_snapshot_time = latest_snapshot.get("last_timestamp")
        if not last_snapshot_time:
            # Fallback: if no last_timestamp in snapshot, create new one
            db.save_state_snapshot(user_id, timestamp)
            logger.debug(f"Created periodic snapshot for user {user_id} (no timestamp in last snapshot)")
            return True

        # Ensure last_snapshot_time is a datetime
        if isinstance(last_snapshot_time, str):
            last_snapshot_time = datetime.fromisoformat(
                last_snapshot_time.replace("Z", "+00:00")
            )

        # Calculate hours since last snapshot
        hours_since = (timestamp - last_snapshot_time).total_seconds() / 3600

        # Create snapshot if interval elapsed
        if hours_since >= snapshot_interval_hours:
            db.save_state_snapshot(user_id, timestamp)
            logger.debug(
                f"Created periodic snapshot for user {user_id} "
                f"({hours_since:.1f} hours since last)"
            )
            return True

        return False

    except Exception as e:
        logger.warning(f"Failed to create periodic snapshot for {user_id}: {e}")
        return False


def process_measurement(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    config: Dict[str, Any],
    unit: str = "kg",
    db=None,
    user_height_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process a single weight measurement through the complete pipeline.

    Single function that:
    1. Cleans and validates data
    2. Manages Kalman state
    3. Applies filtering
    4. Returns comprehensive result

    Args:
        user_id: User identifier
        weight: Weight measurement value
        timestamp: Measurement timestamp
        source: Data source identifier
        config: Combined configuration dictionary
        unit: Unit of measurement
        db: Optional database instance

    Returns:
        Complete processing result with all metadata
    """
    # Ensure weight is a float (handle Decimal from DynamoDB)
    if hasattr(weight, "is_finite"):  # Check if it's a Decimal
        weight = float(weight)
    elif not isinstance(weight, (int, float)):
        weight = float(weight)

    if db is None:
        db = get_state_db()

    # Log input header
    _log("=" * 80)
    _log(f"Processing measurement for user: {user_id[:12]}...")
    _log(f"Weight: {_format_num(weight)}, Unit: {unit}, Timestamp: {timestamp.isoformat()}, Source: {source}")

    # Step 1: Data cleaning and preprocessing
    _log("Step 1: Data cleaning and preprocessing")
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit, user_height_m
    )

    if cleaned_weight is not None:
        _log(f"Cleaned weight: {_format_num(cleaned_weight)}")
    else:
        _log(f"Preprocessing rejected: {preprocess_metadata.get('rejected', 'Unknown reason')}")

    # If preprocessing rejected the measurement
    if cleaned_weight is None:
        _log("Result: REJECTED")
        _log(f"  stage: preprocessing")
        _log("=" * 80)
        return {
            "accepted": False,
            "rejected": True,
            "timestamp": timestamp,
            "source": source,
            "raw_weight": weight,
            "reason": preprocess_metadata.get("rejected", "Preprocessing failed"),
            "stage": "preprocessing",
            "metadata": preprocess_metadata,
        }

    # Step 2: Load or create user state
    _log("Step 2: Load or create user state")
    state = db.get_state(user_id)
    if state is None:
        _log("Creating new state (no existing state)")
        state = db.create_initial_state()
    else:
        # Ensure all numeric values from DynamoDB are proper Python types
        state = ensure_numeric_types(state)
        _log("State exists")
        if "last_raw_weight" in state:
            _log(f"  last_raw_weight: {_format_num(state.get('last_raw_weight'))}")
        if "last_timestamp" in state:
            last_ts = state["last_timestamp"]
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)
            _log(f"  last_timestamp: {last_ts.isoformat()}")
        if "kalman_params" in state:
            _log(f"  kalman_params: present")

    # Use provided height or default
    user_height = user_height_m if user_height_m is not None else PHYSIOLOGICAL_LIMITS["DEFAULT_HEIGHT_M"]

    # Step 3: Check for any type of reset using ResetManager
    _log("Step 3: Check for reset")
    kalman_config = config.get("kalman", {})

    # Check if reset is needed (only if reset features are enabled)
    reset_type = ResetManager.should_trigger_reset(
        state, cleaned_weight, timestamp, source, config
    )

    reset_event = None
    reset_occurred = False

    if reset_type:
        _log(f"Reset needed: type={reset_type}")
        # Perform the reset with transaction safety
        state, reset_event, reset_occurred = _handle_reset_with_transaction(
            user_id, state, reset_type, timestamp, cleaned_weight, source, config
        )
        if reset_occurred and reset_event:
            _log(f"  Reset completed: reason={reset_event.get('reason', 'unknown')}, gap_days={reset_event.get('gap_days', 0)}")
    else:
        _log("No reset needed")

    # Step 4: Initialize Kalman if needed
    _log("Step 4: Initialize Kalman if needed")
    kalman_already_updated = False
    result = None

    if not state.get("kalman_params"):
        _log("Initializing Kalman filter")
        # Check if this is a post-reset initialization
        # For initial measurements, treat current timestamp as "reset" to get adaptive params
        reset_timestamp = get_reset_timestamp(state) if reset_occurred else timestamp

        # Get adaptive Kalman config if within post-reset period
        adaptive_kalman_config = get_adaptive_kalman_params(
            reset_timestamp, timestamp, kalman_config, adaptive_days=7, state=state
        )
        _log(f"  Using adaptive parameters: Q_weight={_format_num(adaptive_kalman_config.get('transition_covariance_weight'))}, Q_trend={_format_num(adaptive_kalman_config.get('transition_covariance_trend'))}")

        # Get adaptive noise for this source
        adaptive_config = config.get("adaptive_noise", {})
        default_multiplier = adaptive_config.get("default_multiplier", 1.5)
        noise_multiplier = get_noise_multiplier(source)
        observation_covariance = (
            adaptive_kalman_config.get("observation_covariance", 3.49)
            * noise_multiplier
        )
        _log(f"  noise_multiplier: {_format_num(noise_multiplier)}")
        _log(f"  observation_covariance: {_format_num(observation_covariance)}")

        kalman_state = KalmanFilterManager.initialize_immediate(
            cleaned_weight, timestamp, adaptive_kalman_config, observation_covariance
        )
        _log(f"  Initial state: {_format_vec(kalman_state.get('last_state'))}")
        # Merge Kalman state with existing state to preserve reset parameters
        state.update(kalman_state)

        # DO NOT call update_state here - initialize_immediate already set the state
        # with the first measurement. Calling update_state would process it twice!

        result = KalmanFilterManager.create_result(
            state, cleaned_weight, timestamp, source, True, observation_covariance
        )

        # Add metadata
        result["stage"] = "initialization"
        result["preprocessing"] = preprocess_metadata
        result["noise_multiplier"] = noise_multiplier

        # Initial measurement - Kalman filter initialized

        # Add reset event info if it occurred (flattened for visualization)
        if reset_occurred:
            result["was_reset"] = True
            result["reset_reason"] = reset_event.get("reason", "unknown")
            result["reset_type"] = reset_event.get("type", "unknown")
            result["gap_days"] = reset_event.get("gap_days", 0)
            # Also keep nested structure for backward compatibility
            result["reset_event"] = {
                "type": reset_event.get("type", "unknown"),
                "gap_days": reset_event.get("gap_days"),
                "reason": reset_event.get("reason", "unknown"),
            }

        # Mark that we've already done the Kalman update
        kalman_already_updated = True

        # Continue to quality validation - no early return during initialization

    # Step 5: Quality scoring (replaces physiological validation)
    _log("Step 5: Quality scoring")
    processing_config = config.get("processing", {})
    quality_config = config.get("quality_scoring", {})

    # Get previous weight and time diff
    previous_weight = None
    time_diff_hours = None

    # Try to get previous weight from Kalman state
    if state:
        current_weight, _ = KalmanFilterManager.get_current_state_values(state)
        if current_weight is not None:
            previous_weight = current_weight
        elif "last_raw_weight" in state:
            previous_weight = ensure_float(state["last_raw_weight"])

        # Get time diff
        if "last_timestamp" in state:
            prev_time = state["last_timestamp"]
            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time)
            time_diff_hours = (timestamp - prev_time).total_seconds() / 3600

    # Get recent weights for statistical analysis
    recent_weights = []
    if state and "measurement_history" in state:
        history = state["measurement_history"]
        if isinstance(history, list):
            recent_weights = [
                ensure_float(h["weight"]) for h in history[-20:] if "weight" in h
            ]

    # Use unified Kalman-centric quality scorer
    # Get Kalman prediction using proper predict step
    kalman_prediction = None
    innovation_covariance = None

    if state and "kalman_params" in state:
        # Use the proper Kalman predict step to get prediction BEFORE update
        kalman_prediction, innovation_covariance = (
            KalmanFilterManager.predict_next_state(state, timestamp)
        )

        # Apply source-specific noise multiplier to innovation covariance if needed
        if innovation_covariance is not None:
            # The predict_next_state already includes base observation noise
            # We need to adjust for source-specific multiplier
            noise_multiplier = get_noise_multiplier(source)
            if noise_multiplier != 1.0:
                # Adjust innovation covariance for source reliability
                # Remove base R, apply multiplier, add back
                kalman_params = state.get("kalman_params", {})
                base_obs_cov = kalman_params.get(
                    "observation_covariance",
                    [[KALMAN_DEFAULTS["observation_covariance"]]],
                )[0][0]
                # Ensure base_obs_cov is float (not Decimal from DB)
                base_obs_cov = float(base_obs_cov)
                # innovation_cov = P_pred[0,0] + R
                # We need: P_pred[0,0] + (R * multiplier)
                predicted_cov_00 = innovation_covariance - base_obs_cov
                innovation_covariance = predicted_cov_00 + (
                    base_obs_cov * noise_multiplier
                )

    # Get recent timestamps if available
    recent_timestamps = []
    if state and "measurement_history" in state:
        history = state["measurement_history"]
        if isinstance(history, list):
            recent_timestamps = [
                h.get("timestamp") for h in history[-20:] if "timestamp" in h
            ]
            # Convert strings to datetime if needed
            recent_timestamps = [
                datetime.fromisoformat(ts) if isinstance(ts, str) else ts
                for ts in recent_timestamps
            ]

    # Create unified scorer instance
    unified_scorer = UnifiedQualityScorer(config=quality_config)

    # Add current timestamp to kalman_state for time-based decay calculation
    kalman_state_with_timestamp = state.copy() if state else {}
    kalman_state_with_timestamp["current_timestamp"] = timestamp

    # Log quality scoring inputs
    if kalman_prediction is not None:
        _log(f"  kalman_prediction: {_format_num(kalman_prediction)}")
    if innovation_covariance is not None:
        _log(f"  innovation_covariance: {_format_num(innovation_covariance)}")
    if previous_weight is not None:
        _log(f"  previous_weight: {_format_num(previous_weight)}")
    if time_diff_hours is not None:
        _log(f"  time_diff_hours: {_format_num(time_diff_hours)}")

    # Calculate quality score
    quality_score = unified_scorer.calculate_quality_score(
        weight=cleaned_weight,
        source=source,
        kalman_state=kalman_state_with_timestamp,
        kalman_prediction=kalman_prediction,
        innovation_covariance=innovation_covariance,
        previous_weight=previous_weight,
        time_diff_hours=time_diff_hours,
        recent_weights=recent_weights,
        recent_timestamps=recent_timestamps,
        user_height_m=user_height,
    )

    _log(f"  quality_score.overall: {_format_num(quality_score.overall)}")
    _log(f"  quality_components: {quality_score.components}")

    if not quality_score.accepted:
        _log(f"  Rejected by quality scorer: {quality_score.rejection_reason}")
        _log("Result: REJECTED")
        _log(f"  stage: unified_quality_scoring")
        _log("=" * 80)
        return {
            "accepted": False,
            "timestamp": timestamp,
            "raw_weight": weight,
            "cleaned_weight": cleaned_weight,
            "source": source,
            "reason": quality_score.rejection_reason,
            "stage": "unified_quality_scoring",
            "quality_score": quality_score.overall,
            "quality_components": quality_score.components,
            "quality_details": quality_score.to_dict(),
        }

    # Store quality score for later use
    quality_score_value = quality_score.overall
    quality_components = quality_score.components

    # Only do Kalman update if not already done during initialization
    if not kalman_already_updated:
        _log("Step 6: Kalman update")
        # Check if we should use adaptive parameters (within 7 days of reset)
        reset_timestamp = get_reset_timestamp(state)
        adaptive_kalman_config = get_adaptive_kalman_params(
            reset_timestamp, timestamp, kalman_config, adaptive_days=7, state=state
        )

        # Update state's kalman_params with adaptive values
        if (
            reset_timestamp
            and (timestamp - reset_timestamp).total_seconds() / 86400.0 < 7
        ):
            # Only update if we have the adaptive values
            if (
                "transition_covariance_weight" in adaptive_kalman_config
                and "transition_covariance_trend" in adaptive_kalman_config
            ):
                state["kalman_params"]["transition_covariance"] = [
                    [adaptive_kalman_config["transition_covariance_weight"], 0],
                    [0, adaptive_kalman_config["transition_covariance_trend"]],
                ]

        adaptive_config = config.get("adaptive_noise", {})
        default_multiplier = adaptive_config.get("default_multiplier", 1.5)
        noise_multiplier = get_noise_multiplier(source)
        observation_covariance = (
            adaptive_kalman_config.get("observation_covariance", 3.49)
            * noise_multiplier
        )
        _log(f"  Using adaptive parameters: Q_weight={_format_num(adaptive_kalman_config.get('transition_covariance_weight'))}, Q_trend={_format_num(adaptive_kalman_config.get('transition_covariance_trend'))}")
        _log(f"  observation_covariance: {_format_num(observation_covariance)}")

        # Apply trend limiting before update
        current_weight, current_trend = KalmanFilterManager.get_current_state_values(
            state
        )
        if state.get("last_state") is not None:
            _log(f"  State before update: {_format_vec(state.get('last_state'))}")
        if current_trend is not None:
            _log(f"  Trend before limiting: {_format_num(current_trend)}")
            # Limit trend to ±5kg/week (±0.714kg/day)
            max_daily_trend = 0.714  # 5kg/week
            if abs(current_trend) > max_daily_trend:
                # Clamp the trend in the state before update
                limited_trend = (
                    max_daily_trend if current_trend > 0 else -max_daily_trend
                )
                if state.get("last_state") is not None:
                    last_state = state["last_state"]
                    if len(last_state.shape) > 1:
                        last_state[-1][1] = limited_trend
                    else:
                        last_state[1] = limited_trend

        state = KalmanFilterManager.update_state(
            state, cleaned_weight, timestamp, source, {}, observation_covariance
        )

        if state.get("last_state") is not None:
            _log(f"  State after update: {_format_vec(state.get('last_state'))}")

        # Apply trend limiting after update
        current_weight, current_trend = KalmanFilterManager.get_current_state_values(
            state
        )
        if current_trend is not None:
            _log(f"  Trend after update (before limiting): {_format_num(current_trend)}")
        if current_trend is not None and abs(current_trend) > 0.714:
            _log(f"  Limiting trend from {_format_num(current_trend)} to ±0.714")
            # Clamp the trend after update
            limited_trend = 0.714 if current_trend > 0 else -0.714
            if state.get("last_state") is not None:
                last_state = state["last_state"]
                if len(last_state.shape) > 1:
                    last_state[-1][1] = limited_trend
                else:
                    last_state[1] = limited_trend

        result = KalmanFilterManager.create_result(
            state, cleaned_weight, timestamp, source, True, observation_covariance
        )

    # Step 8: Add comprehensive metadata
    result["preprocessing"] = preprocess_metadata
    # Set noise multiplier if not already set
    if "noise_multiplier" not in result:
        result["noise_multiplier"] = noise_multiplier
    result["stage"] = "accepted"

    # Add quality score if available
    # Always add quality score for accepted measurements
    result["quality_score"] = quality_score_value
    result["quality_components"] = quality_components

    # Add reset event info if it occurred
    if reset_occurred:
        result["reset_event"] = {
            "type": reset_event.get("type", "unknown"),
            "gap_days": reset_event.get("gap_days"),
            "reason": reset_event.get("reason", "unknown"),
        }

    # Calculate BMI details
    implied_bmi = cleaned_weight / (user_height**2)
    result["bmi_details"] = {
        "user_height_m": user_height,
        "implied_bmi": round(implied_bmi, 1),
        "original_weight": weight,
        "original_unit": unit,
        "cleaned_weight": cleaned_weight,
    }

    # Update measurement history for quality scoring
    if "measurement_history" not in state:
        state["measurement_history"] = []

    state["measurement_history"].append(
        {
            "weight": cleaned_weight,
            "timestamp": timestamp.isoformat(),
            "quality_score": quality_score_value,
            "source": source,
        }
    )

    # Keep only last 30 measurements
    state["measurement_history"] = state["measurement_history"][-30:]

    # Save updated state - Main successful processing path
    # Increment measurements counter for adaptation tracking
    state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1
    state["last_source"] = source
    state["last_timestamp"] = timestamp  # Keep for backward compatibility
    state["last_accepted_timestamp"] = timestamp
    state["last_raw_weight"] = cleaned_weight  # Track for soft reset detection

    # Update temporal baseline for continuous tracking
    unified_scorer = UnifiedQualityScorer(config=config)
    state = unified_scorer.update_temporal_baseline(state, cleaned_weight, timestamp)

    # Validate state before persistence
    is_valid, error_msg = PersistenceValidator.validate_state(
        state, user_id, reason="successful_processing"
    )
    if is_valid:
        # Get previous state for change detection
        previous_state = db.get_state(user_id)
        should_persist, audit_msg = PersistenceValidator.should_persist(
            state, previous_state, user_id, reason="successful_processing"
        )

        if should_persist:
            db.save_state(user_id, state)
            PersistenceValidator.create_audit_log(
                user_id,
                "persist",
                state,
                True,
                reason="successful_processing",
                error=None,
            )

            # Save snapshot after reset for replay functionality
            if reset_occurred:
                try:
                    db.save_state_snapshot(user_id, timestamp)
                except Exception as e:
                    logger.warning(
                        f"Failed to save post-reset snapshot for {user_id}: {e}"
                    )
                    # Continue processing even if snapshot fails

            # Create periodic snapshot if interval has elapsed
            # This ensures replay has adequate state history even without resets
            _maybe_create_periodic_snapshot(db, user_id, timestamp, config)
        else:
            # Log why we're not persisting
            PersistenceValidator.create_audit_log(
                user_id, "skip", state, True, reason=audit_msg, error=None
            )
    else:
        # Log validation failure
        PersistenceValidator.create_audit_log(
            user_id,
            "validate_failed",
            state,
            False,
            reason="successful_processing",
            error=error_msg,
        )

    # if result.get("quality_score") < 0.5:
    #     result["warning"] = "Low quality score - review measurement history"
    #     result["accepted"] = False  # Mark as not accepted for downstream handling

    # Log final result
    _log(f"Result: {'ACCEPTED' if result.get('accepted', True) else 'REJECTED'}")
    if result.get("kalman_estimate") is not None:
        _log(f"  kalman_estimate: {_format_num(result.get('kalman_estimate'))}")
    if result.get("kalman_uncertainty") is not None:
        _log(f"  kalman_uncertainty: {_format_num(result.get('kalman_uncertainty'))}")
    if result.get("quality_score") is not None:
        _log(f"  quality_score: {_format_num(result.get('quality_score'))}")
    _log(f"  stage: {result.get('stage', 'unknown')}")
    _log("=" * 80)

    return result


# Backward compatibility wrapper
def process_weight_enhanced(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    processing_config: Dict,
    kalman_config: Dict,
    unit: str = "kg",
) -> Optional[Dict]:
    """Backward compatibility wrapper for process_measurement."""
    config = {"processing": processing_config, "kalman": kalman_config}

    # Handle nested config for adaptive noise
    if "config" in processing_config:
        config.update(processing_config["config"])

    return process_measurement(user_id, weight, timestamp, source, config, unit)


# Circuit breaker for reset operations (module level)
_reset_circuit_breaker = CircuitBreaker(
    failure_threshold=3, timeout=60, success_threshold=2, name="reset_operations"
)


def _handle_reset_with_transaction(
    user_id: str,
    state: Dict[str, Any],
    reset_type: ResetType,
    timestamp: datetime,
    weight: float,
    source: str,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict], bool]:
    """
    Handle reset operations with transaction safety and circuit breaker.

    Args:
        user_id: User identifier
        state: Current state
        reset_type: Type of reset to perform
        timestamp: Measurement timestamp
        weight: Weight value
        source: Data source
        config: Configuration

    Returns:
        Tuple of (new_state, reset_event, reset_occurred)
    """
    try:
        # Try through circuit breaker first
        return _reset_circuit_breaker.call(
            _perform_transactional_reset,
            user_id,
            state,
            reset_type,
            timestamp,
            weight,
            source,
            config,
        )
    except CircuitOpenError as e:
        logger.error(f"Reset circuit open for user {user_id}: {e}")
        # Return original state without reset
        return state, None, False
    except Exception as e:
        logger.error(f"Reset failed for user {user_id}: {e}")
        # Return original state without reset
        return state, None, False


def _perform_transactional_reset(
    user_id: str,
    state: Dict[str, Any],
    reset_type: ResetType,
    timestamp: datetime,
    weight: float,
    source: str,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict], bool]:
    """
    Perform reset with transaction management.

    Returns:
        Tuple of (new_state, reset_event, reset_occurred)
    """
    with ResetTransaction(user_id) as txn:
        # Save original state for potential rollback
        txn.save_original_state(ResetOperation.STATE_UPDATE, state)

        try:
            # Step 1: Perform the actual reset
            # Handle both ResetType enum and string
            reset_type_value = (
                reset_type.value if hasattr(reset_type, "value") else reset_type
            )
            new_state, reset_event = ResetManager.perform_reset(
                state, reset_type, timestamp, weight, source, config
            )

            # Save checkpoint and validate
            txn.save_checkpoint(ResetOperation.STATE_UPDATE, new_state)
            if not txn.validate_checkpoint(ResetOperation.STATE_UPDATE):
                raise ValueError(
                    f"State validation failed after {reset_type_value} reset"
                )

            txn.mark_completed(ResetOperation.STATE_UPDATE)

            # Step 2: Validate Kalman reset (kalman_params should be None)
            kalman_state = {
                "kalman_params": new_state.get("kalman_params"),
                "reset_parameters": new_state.get("reset_parameters"),
                "measurements_since_reset": new_state.get(
                    "measurements_since_reset", 0
                ),
                "reset_type": new_state.get("reset_type"),
                "reset_timestamp": new_state.get("reset_timestamp"),
            }

            txn.save_checkpoint(ResetOperation.KALMAN_RESET, kalman_state)
            if not txn.validate_checkpoint(ResetOperation.KALMAN_RESET):
                raise ValueError("Kalman state validation failed after reset")

            txn.mark_completed(ResetOperation.KALMAN_RESET)

            # All operations succeeded
            logger.info(f"Reset transaction completed successfully for user {user_id}")
            return new_state, reset_event, True

        except Exception as e:
            import traceback

            logger.error(f"Reset transaction failed for user {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Transaction will automatically rollback
            # Return original state
            original_state = txn.get_original_state(ResetOperation.STATE_UPDATE)
            if original_state:
                return original_state, None, False
            return state, None, False
