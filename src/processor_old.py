"""
Enhanced weight processor with data quality improvements.
Orchestrates validation, quality, and Kalman filtering components.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, List
from collections import defaultdict, deque
import numpy as np

try:
    from .database import ProcessorStateDB, get_state_db
    from .kalman import KalmanFilterManager
    from .validation import PhysiologicalValidator, BMIValidator, ThresholdCalculator
    from .quality import (
        DataQualityPreprocessor,
        AdaptiveOutlierDetector,
        AdaptiveKalmanConfig,
        quality_monitor
    )
    from .models import (
        QUESTIONNAIRE_SOURCES,
        categorize_rejection_enhanced,
        get_rejection_severity,
        KALMAN_DEFAULTS
    )
except ImportError:
    from database import ProcessorStateDB, get_state_db
    from kalman import KalmanFilterManager
    from validation import PhysiologicalValidator, BMIValidator, ThresholdCalculator
    from quality import (
        DataQualityPreprocessor,
        AdaptiveOutlierDetector,
        AdaptiveKalmanConfig,
        quality_monitor
    )
    from models import (
        QUESTIONNAIRE_SOURCES,
        categorize_rejection_enhanced,
        get_rejection_severity,
        KALMAN_DEFAULTS
    )


def get_adaptive_noise_multiplier(source: str, config: Dict) -> float:
    """Get noise multiplier from config for a given source."""
    adaptive_config = config.get('adaptive_noise', {})
    
    if not adaptive_config.get('enabled', False):
        return 1.0
    
    multipliers = adaptive_config.get('multipliers', {})
    default_multiplier = adaptive_config.get('default_multiplier', 1.5)
    
    return multipliers.get(source, default_multiplier)


class WeightProcessor:
    """
    Stateless weight processor - ALL values processed immediately.
    No buffering, no waiting for initialization - processes everything.
    """

    @staticmethod
    def process_weight(
        user_id: str,
        weight: float,
        timestamp: datetime,
        source: str,
        processing_config: Dict,
        kalman_config: Dict,
        db=None,
        observation_covariance: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Process a single weight measurement for a user.
        
        ALWAYS returns a result - no buffering, no waiting.
        After long gaps (>30 days), resets state for fresh start.
        
        Args:
            user_id: User identifier
            weight: Weight measurement in kg
            timestamp: Measurement timestamp
            source: Data source identifier
            processing_config: Processing configuration dict
            kalman_config: Kalman filter configuration dict
            db: Optional database instance (creates new if None)
            
        Returns:
            Result dictionary - NEVER None, all measurements processed
        """
        if db is None:
            db = get_state_db()
        
        state = db.get_state(user_id)
        if state is None:
            state = db.create_initial_state()
        
        result, updated_state = WeightProcessor._process_weight_internal(
            weight, timestamp, source, state, processing_config, kalman_config, user_id, observation_covariance
        )
        
        if updated_state:
            updated_state['last_source'] = source
            # Save adapted parameters if present in result
            if result.get('adapted_params'):
                updated_state['adapted_params'] = result['adapted_params']
            updated_state['last_attempt_timestamp'] = timestamp
            if not result.get('accepted', False):
                updated_state['rejection_count_since_accept'] = updated_state.get('rejection_count_since_accept', 0) + 1
            else:
                updated_state['rejection_count_since_accept'] = 0
            db.save_state(user_id, updated_state)
        else:
            if state:
                state['last_attempt_timestamp'] = timestamp
                state['rejection_count_since_accept'] = state.get('rejection_count_since_accept', 0) + 1
                db.save_state(user_id, state)
        
        return result

    @staticmethod
    def _process_weight_internal(
        weight: float,
        timestamp: datetime,
        source: str,
        state: Dict[str, Any],
        processing_config: dict,
        kalman_config: dict,
        user_id: Optional[str] = None,
        observation_covariance: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Internal processing logic - pure functional.
        
        Returns:
            Tuple of (result, updated_state)
            Result is never None - all measurements are processed
        """
        new_state = state.copy()
        
        # Migrate old state format to include new fields
        if 'last_attempt_timestamp' not in new_state and 'last_timestamp' in new_state:
            new_state['last_attempt_timestamp'] = new_state['last_timestamp']
        if 'rejection_count_since_accept' not in new_state:
            new_state['rejection_count_since_accept'] = 0
        
        # Get user height and add to processing config
        if user_id:
            user_height = DataQualityPreprocessor.get_user_height(user_id)
            processing_config = processing_config.copy()
            processing_config['user_height_m'] = user_height
        
        if not state.get('kalman_params'):
            # Use passed observation_covariance if provided, otherwise use config value
            obs_cov = observation_covariance if observation_covariance is not None else kalman_config.get('observation_covariance', KALMAN_DEFAULTS['observation_covariance'])
            
            new_state = KalmanFilterManager.initialize_immediate(
                weight, timestamp, kalman_config, obs_cov
            )
            
            new_state = KalmanFilterManager.update_state(
                new_state, weight, timestamp, source, processing_config, obs_cov
            )
            
            result = KalmanFilterManager.create_result(
                new_state, weight, timestamp, source, True, obs_cov
            )
            return result, new_state
        
        time_delta_days = KalmanFilterManager.calculate_time_delta_days(
            timestamp, new_state.get('last_timestamp')
        )
        
        last_attempt = state.get('last_attempt_timestamp', state.get('last_timestamp'))
        attempt_gap_days = KalmanFilterManager.calculate_time_delta_days(
            timestamp, last_attempt
        )
        
        reset_gap_days = kalman_config.get("reset_gap_days", 30)
        
        if state.get('last_source') in QUESTIONNAIRE_SOURCES:
            reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)
        
        if attempt_gap_days > reset_gap_days:
            height_m = processing_config.get('user_height_m', 1.7)
            min_bmi = processing_config.get('min_valid_bmi', 10.0)
            max_bmi = processing_config.get('max_valid_bmi', 60.0)
            
            is_valid, rejection_reason = BMIValidator.validate_weight_bmi_only(
                weight, height_m, min_bmi, max_bmi
            )
            
            if not is_valid:
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "accepted": False,
                    "reason": f"Post-gap BMI validation failed: {rejection_reason}",
                    "source": source,
                    "gap_days": time_delta_days,
                    "was_gap_reset_attempted": True
                }, None
            
            # Use passed observation_covariance if provided
            obs_cov = observation_covariance if observation_covariance is not None else kalman_config.get('observation_covariance', KALMAN_DEFAULTS['observation_covariance'])
            
            new_state = KalmanFilterManager.initialize_immediate(
                weight, timestamp, kalman_config, obs_cov
            )
            new_state = KalmanFilterManager.update_state(
                new_state, weight, timestamp, source, processing_config, obs_cov
            )
            result = KalmanFilterManager.create_result(
                new_state, weight, timestamp, source, True, obs_cov
            )
            result['was_reset'] = True
            result['gap_days'] = attempt_gap_days
            result['reset_reason'] = f"Gap reset after {attempt_gap_days:.1f} days of no data"
            result['rejection_count_before_reset'] = state.get('rejection_count_since_accept', 0)
            return result, new_state
        
        is_valid, rejection_reason = PhysiologicalValidator.validate_weight(
            weight, processing_config, state, timestamp
        )
        if not is_valid:
            return {
                "timestamp": timestamp,
                "raw_weight": weight,
                "accepted": False,
                "reason": rejection_reason or "Basic validation failed",
                "source": source,
            }, None
        
        current_weight, current_trend = KalmanFilterManager.get_current_state_values(state)
        
        if current_weight is not None:
            predicted_weight = current_weight + current_trend * time_delta_days
            deviation = abs(weight - predicted_weight) / predicted_weight
            
            extreme_threshold = processing_config["extreme_threshold"]
            
            if deviation > extreme_threshold:
                pseudo_normalized_innovation = (deviation / extreme_threshold) * 3.0
                confidence = KalmanFilterManager.calculate_confidence(
                    pseudo_normalized_innovation
                )
                
                return {
                    "timestamp": timestamp,
                    "raw_weight": weight,
                    "filtered_weight": float(predicted_weight),
                    "trend": float(current_trend),
                    "accepted": False,
                    "reason": f"Extreme deviation: {deviation:.1%}",
                    "confidence": confidence,
                    "source": source,
                }, None
        
        # Use passed observation_covariance if provided
        obs_cov = observation_covariance if observation_covariance is not None else kalman_config.get('observation_covariance', KALMAN_DEFAULTS['observation_covariance'])
        
        new_state = KalmanFilterManager.update_state(
            new_state, weight, timestamp, source, processing_config, obs_cov
        )
        
        result = KalmanFilterManager.create_result(
            new_state, weight, timestamp, source, True, obs_cov
        )
        
        return result, new_state

    @staticmethod
    def get_user_state(user_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state for a user (for debugging/inspection)."""
        db = get_state_db()
        return db.get_state(user_id)

    @staticmethod
    def reset_user(user_id: str) -> bool:
        """Reset a user's state (delete from database)."""
        db = get_state_db()
        return db.delete_state(user_id)


def process_weight_enhanced(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    processing_config: Dict,
    kalman_config: Dict,
    unit: str = 'kg'
) -> Optional[Dict]:
    """
    Enhanced weight processing with data quality improvements.
    
    This wraps the original WeightProcessor with additional defensive layers:
    1. Pre-processing for unit conversion and data cleaning
    2. Adaptive outlier thresholds based on source reliability
    3. Kalman noise adaptation based on source quality
    4. Real-time quality monitoring
    
    Args:
        user_id: User identifier
        weight: Weight measurement value
        timestamp: Measurement timestamp
        source: Data source identifier
        processing_config: Processing configuration
        kalman_config: Kalman filter configuration
        unit: Unit of measurement ('kg', 'lb', 'lbs', 'pound', 'pounds', etc.)
        
    Returns:
        Processing result with additional metadata, or None if rejected
    """
    
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp, user_id, unit
    )
    
    if cleaned_weight is None:
        return {
            'rejected': True,
            'accepted': False,
            'stage': 'preprocessing',
            'reason': preprocess_metadata.get('rejected'),
            'rejection_reason': preprocess_metadata.get('rejected'),
            'metadata': preprocess_metadata,
            'timestamp': timestamp,
            'source': source,
            'raw_weight': weight,
            'original_weight': weight,
            'original_unit': unit,
            'bmi_details': {
                'detected_as_bmi': 'BMI' in preprocess_metadata.get('rejected', ''),
                'user_height_m': preprocess_metadata.get('user_height_m'),
                'implied_bmi': preprocess_metadata.get('implied_bmi'),
                'bmi_category': preprocess_metadata.get('bmi_category')
            }
        }
    
    db = get_state_db()
    state = db.get_state(user_id)
    
    time_gap_days = 0
    if state and state.get('last_timestamp'):
        time_gap_days = (timestamp - state['last_timestamp']).days
    
    adapted_config = processing_config.copy()
    
    threshold_result = ThresholdCalculator.get_extreme_deviation_threshold(
        source=source,
        time_gap_days=time_gap_days,
        current_weight=cleaned_weight,
        unit='percentage'
    )
    
    adapted_config['extreme_threshold'] = threshold_result.value
    adapted_config['extreme_threshold_pct'] = threshold_result.value
    adapted_config['extreme_threshold_kg'] = threshold_result.metadata.get('absolute_threshold_kg')
    
    adapted_kalman = kalman_config.copy()
    
    # Get noise multiplier from config if adaptive noise is enabled
    config = processing_config.get('config', {})
    adaptive_config = config.get('adaptive_noise', {})
    
    if adaptive_config.get('enabled', False):
        multipliers = adaptive_config.get('multipliers', {})
        default_multiplier = adaptive_config.get('default_multiplier', 1.5)
        noise_multiplier = multipliers.get(source, default_multiplier)
    else:
        # Fall back to old method if adaptive noise is disabled
        noise_multiplier = ThresholdCalculator.get_measurement_noise_multiplier(source)
    
    base_noise = kalman_config.get('observation_covariance', KALMAN_DEFAULTS['observation_covariance'])
    adapted_observation_covariance = base_noise * noise_multiplier
    adapted_kalman['observation_covariance'] = adapted_observation_covariance
    
    # Track adapted parameters if they differ from base
    adapted_params = None
    if noise_multiplier != 1.0:
        adapted_params = {
            'process_noise': adapted_kalman.get('transition_covariance_weight', kalman_config.get('transition_covariance_weight')),
            'measurement_noise': adapted_kalman['observation_covariance'],
            'noise_multiplier': noise_multiplier,
            'source': source
        }
    
    if noise_multiplier > 2.0:
        adapted_kalman['initial_variance'] = kalman_config.get('initial_variance', 1.0) * 1.5
    
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=cleaned_weight,
        timestamp=timestamp,
        source=source,
        processing_config=adapted_config,
        kalman_config=adapted_kalman,
        observation_covariance=adapted_observation_covariance
    )
    
    # Add adapted parameters to result if they exist
    if adapted_params:
        result['adapted_params'] = adapted_params
    
    if result:
        is_outlier = False
        outlier_reason = None
        if state and state.get('last_state') is not None:
            last_state = state['last_state']
            try:
                if isinstance(last_state, (list, tuple, np.ndarray)):
                    last_weight = float(last_state[0])
                else:
                    last_weight = float(last_state)
                
                weight_change = abs(cleaned_weight - last_weight)
                is_outlier, outlier_reason = AdaptiveOutlierDetector.check_outlier(
                    weight_change, source, time_gap_days
                )
            except (IndexError, TypeError, ValueError):
                pass
        
        is_rejected = result.get('rejected', False) or not result.get('accepted', True)
        alert = quality_monitor.record_measurement(source, is_outlier, is_rejected)
        
        result['preprocessing_metadata'] = preprocess_metadata
        
        result['threshold_info'] = {
            'extreme_threshold_pct': adapted_config.get('extreme_threshold_pct'),
            'extreme_threshold_kg': adapted_config.get('extreme_threshold_kg'),
            'source_reliability': ThresholdCalculator.get_source_reliability(source),
            'measurement_noise_multiplier': noise_multiplier
        }
        
        result['adaptive_threshold'] = adapted_config['extreme_threshold']
        result['measurement_noise_used'] = adapted_kalman.get('observation_covariance', 1.0)
        
        user_height = DataQualityPreprocessor.get_user_height(user_id)
        implied_bmi = round(cleaned_weight / (user_height ** 2), 1)
        
        if implied_bmi < 18.5:
            bmi_category = 'underweight'
        elif implied_bmi < 25:
            bmi_category = 'normal'
        elif implied_bmi < 30:
            bmi_category = 'overweight'
        else:
            bmi_category = 'obese'
        
        result['bmi_details'] = {
            'user_height_m': user_height,
            'original_weight': weight,
            'original_unit': unit,
            'cleaned_weight': cleaned_weight,
            'implied_bmi': implied_bmi,
            'bmi_category': bmi_category,
            'bmi_converted': weight != cleaned_weight and 15 <= weight <= 50,
            'unit_converted': any('Converted' in c for c in preprocess_metadata.get('corrections', [])),
            'corrections': preprocess_metadata.get('corrections', []),
            'warnings': preprocess_metadata.get('warnings', [])
        }
        
        if result.get('accepted') == False:
            rejection_reason = result.get('reason', '')
            result['rejection_insights'] = {
                'category': categorize_rejection_enhanced(rejection_reason),
                'severity': get_rejection_severity(rejection_reason, weight_change if 'weight_change' in locals() else 0),
                'source_reliability': ThresholdCalculator.get_source_reliability(source),
                'adaptive_threshold_used': adapted_config['extreme_threshold'],
                'outlier_detected': is_outlier,
                'outlier_reason': outlier_reason
            }
        
        if alert:
            result['quality_alert'] = alert
        
        result['source_quality'] = quality_monitor.get_source_summary(source)
    
    # Save adapted parameters to database if they exist
    if adapted_params and result.get("accepted"):
        db = get_state_db()
        state = db.get_state(user_id)
        if state:
            state["adapted_params"] = adapted_params
            db.save_state(user_id, state)
    
    return result


