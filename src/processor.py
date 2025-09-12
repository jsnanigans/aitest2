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
        get_rejection_severity
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
        get_rejection_severity
    )


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
            weight, timestamp, source, state, processing_config, kalman_config
        )
        
        if updated_state:
            updated_state['last_source'] = source
            db.save_state(user_id, updated_state)
        
        return result

    @staticmethod
    def _process_weight_internal(
        weight: float,
        timestamp: datetime,
        source: str,
        state: Dict[str, Any],
        processing_config: dict,
        kalman_config: dict
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Internal processing logic - pure functional.
        
        Returns:
            Tuple of (result, updated_state)
            Result is never None - all measurements are processed
        """
        new_state = state.copy()
        
        if not state.get('kalman_params'):
            new_state = KalmanFilterManager.initialize_immediate(
                weight, timestamp, kalman_config
            )
            
            new_state = KalmanFilterManager.update_state(
                new_state, weight, timestamp, source, processing_config
            )
            
            result = KalmanFilterManager.create_result(
                new_state, weight, timestamp, source, True
            )
            return result, new_state
        
        time_delta_days = KalmanFilterManager.calculate_time_delta_days(
            timestamp, new_state.get('last_timestamp')
        )
        
        reset_gap_days = kalman_config.get("reset_gap_days", 30)
        
        if state.get('last_source') in QUESTIONNAIRE_SOURCES:
            reset_gap_days = kalman_config.get("questionnaire_reset_days", 10)
        
        if time_delta_days > reset_gap_days:
            new_state = KalmanFilterManager.initialize_immediate(
                weight, timestamp, kalman_config
            )
            new_state = KalmanFilterManager.update_state(
                new_state, weight, timestamp, source, processing_config
            )
            result = KalmanFilterManager.create_result(
                new_state, weight, timestamp, source, True
            )
            result['was_reset'] = True
            result['gap_days'] = time_delta_days
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
        
        new_state = KalmanFilterManager.update_state(
            new_state, weight, timestamp, source, processing_config
        )
        
        result = KalmanFilterManager.create_result(
            new_state, weight, timestamp, source, True
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
    
    noise_multiplier = ThresholdCalculator.get_measurement_noise_multiplier(source)
    
    base_noise = kalman_config.get('observation_covariance', 1.0)
    adapted_kalman['observation_covariance'] = base_noise * noise_multiplier
    
    if noise_multiplier > 2.0:
        adapted_kalman['initial_variance'] = kalman_config.get('initial_variance', 1.0) * 1.5
    
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=cleaned_weight,
        timestamp=timestamp,
        source=source,
        processing_config=adapted_config,
        kalman_config=adapted_kalman
    )
    
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
    
    return result


class DynamicResetManager:
    """
    Manages dynamic reset decisions for the weight processor.
    
    Features:
    - Post-questionnaire gap reduction (10 days instead of 30)
    - Variance-based reset for outlier detection
    - Source-adaptive thresholds based on reliability
    - Statistical change point detection
    - Configurable voting mechanism for combined decisions
    """
    
    DEFAULT_CONFIG = {
        'questionnaire_gap_days': 10,
        'standard_gap_days': 30,
        'variance_threshold': 0.15,
        'trend_reversal_threshold': 0.5,
        'change_point_z_score': 3.0,
        'cusum_k_factor': 0.5,
        'cusum_h_factor': 4.0,
        'min_history_for_changepoint': 5,
        'max_history_length': 20,
        'combined_vote_threshold': 2,
        'enable_questionnaire_gap': True,
        'enable_variance_reset': True,
        'enable_reliability_reset': True,
        'enable_changepoint_reset': False,
    }
    
    RELIABILITY_GAPS = {
        'excellent': 45,
        'good': 30,
        'moderate': 20,
        'poor': 15,
        'unknown': 25
    }
    
    QUESTIONNAIRE_SOURCES = [
        'questionnaire',
        'internal-questionnaire',
        'initial-questionnaire',
        'care-team-upload',
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Dynamic Reset Manager.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.reset_history = deque(maxlen=100)
        
    def should_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        method: str = 'auto'
    ) -> Tuple[bool, str, Dict]:
        """
        Determine if state should be reset.
        
        Args:
            current_weight: Current weight measurement (kg)
            timestamp: Current measurement timestamp
            source: Data source identifier
            state: Current processor state
            method: Reset method ('auto', 'questionnaire_gap', 'variance', 
                   'source_reliability', 'change_point', 'combined')
        
        Returns:
            Tuple of (should_reset, reason, metadata)
        """
        
        metadata = {
            'timestamp': timestamp,
            'source': source,
            'weight': current_weight,
            'method': method
        }
        
        if not state or not state.get('last_timestamp'):
            return False, "No previous state", metadata
        
        last_timestamp = state['last_timestamp']
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)
        
        gap_days = (timestamp - last_timestamp).total_seconds() / 86400
        metadata['gap_days'] = gap_days
        
        if method == 'auto':
            if self.config['enable_questionnaire_gap']:
                method = 'questionnaire_gap'
            elif self.config['enable_variance_reset']:
                method = 'variance'
            elif self.config['enable_reliability_reset']:
                method = 'source_reliability'
            else:
                method = 'standard'
        
        if method == 'questionnaire_gap':
            return self._questionnaire_gap_reset(source, gap_days, state, metadata)
        elif method == 'variance':
            return self._variance_reset(current_weight, state, metadata)
        elif method == 'source_reliability':
            return self._reliability_reset(source, gap_days, metadata)
        elif method == 'change_point':
            return self._changepoint_reset(current_weight, state, metadata)
        elif method == 'combined':
            return self._combined_reset(current_weight, timestamp, source, state, gap_days, metadata)
        else:
            if gap_days > self.config['standard_gap_days']:
                reason = f"Standard gap reset ({gap_days:.1f} days)"
                metadata['reset_type'] = 'standard_gap'
                return True, reason, metadata
            return False, "No reset needed", metadata
    
    def _questionnaire_gap_reset(
        self, 
        source: str, 
        gap_days: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Apply questionnaire-specific gap threshold."""
        
        last_source = state.get('last_source', '')
        
        is_post_questionnaire = any(
            q in last_source.lower() 
            for q in self.QUESTIONNAIRE_SOURCES
        )
        
        metadata['last_source'] = last_source
        metadata['is_post_questionnaire'] = is_post_questionnaire
        
        if is_post_questionnaire:
            threshold = self.config['questionnaire_gap_days']
            if gap_days > threshold:
                reason = f"Post-questionnaire reset (gap: {gap_days:.1f}/{threshold} days)"
                metadata['reset_type'] = 'questionnaire_gap'
                metadata['threshold_used'] = threshold
                return True, reason, metadata
        else:
            threshold = self.config['standard_gap_days']
            if gap_days > threshold:
                reason = f"Standard gap reset ({gap_days:.1f}/{threshold} days)"
                metadata['reset_type'] = 'standard_gap'
                metadata['threshold_used'] = threshold
                return True, reason, metadata
        
        return False, f"Within threshold ({gap_days:.1f}/{threshold} days)", metadata
    
    def _variance_reset(
        self, 
        current_weight: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Detect high variance or trend reversals."""
        
        if state.get('last_state') is None:
            return False, "No state history", metadata
        
        last_state = state['last_state']
        if isinstance(last_state, np.ndarray):
            if len(last_state.shape) > 1:
                filtered_weight = last_state[-1][0]
                trend = last_state[-1][1]
            else:
                filtered_weight = last_state[0]
                trend = 0
        else:
            filtered_weight = last_state[0] if isinstance(last_state, (list, tuple)) else last_state
            trend = 0
        
        deviation = abs(current_weight - filtered_weight)
        deviation_pct = deviation / filtered_weight
        
        metadata['filtered_weight'] = filtered_weight
        metadata['deviation'] = deviation
        metadata['deviation_pct'] = deviation_pct
        metadata['trend'] = trend
        
        if deviation_pct > self.config['variance_threshold']:
            reason = f"High variance reset (deviation: {deviation_pct:.1%})"
            metadata['reset_type'] = 'variance'
            return True, reason, metadata
        
        if abs(trend) > self.config['trend_reversal_threshold']:
            weight_change = current_weight - filtered_weight
            if np.sign(weight_change) != np.sign(trend):
                reason = f"Trend reversal (trend: {trend:.2f}, change: {weight_change:.2f})"
                metadata['reset_type'] = 'trend_reversal'
                return True, reason, metadata
        
        return False, "Variance within bounds", metadata
    
    def _reliability_reset(
        self, 
        source: str, 
        gap_days: float,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Apply source reliability-based thresholds."""
        
        reliability = ThresholdCalculator.get_source_reliability(source)
        threshold = self.RELIABILITY_GAPS.get(reliability, 30)
        
        metadata['source_reliability'] = reliability
        metadata['reliability_threshold'] = threshold
        
        if gap_days > threshold:
            reason = f"Source-adaptive reset ({reliability}, gap: {gap_days:.1f}/{threshold} days)"
            metadata['reset_type'] = 'source_reliability'
            return True, reason, metadata
        
        return False, f"Within {reliability} threshold ({gap_days:.1f}/{threshold} days)", metadata
    
    def _changepoint_reset(
        self, 
        current_weight: float, 
        state: Dict,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Statistical change point detection using CUSUM."""
        
        if 'measurement_history' not in state:
            state['measurement_history'] = []
        
        history = state['measurement_history']
        history.append(current_weight)
        
        if len(history) > self.config['max_history_length']:
            history = history[-self.config['max_history_length']:]
            state['measurement_history'] = history
        
        metadata['history_length'] = len(history)
        
        if len(history) < self.config['min_history_for_changepoint']:
            return False, "Insufficient history", metadata
        
        recent_history = history[:-1]
        mean = np.mean(recent_history)
        std = np.std(recent_history) if len(recent_history) > 1 else 1.0
        
        metadata['history_mean'] = mean
        metadata['history_std'] = std
        
        if std > 0:
            z_score = abs((current_weight - mean) / std)
            metadata['z_score'] = z_score
            
            if z_score > self.config['change_point_z_score']:
                reason = f"Change point detected (z-score: {z_score:.2f})"
                metadata['reset_type'] = 'change_point'
                state['measurement_history'] = [current_weight]
                return True, reason, metadata
        
        if len(history) >= 10:
            k = self.config['cusum_k_factor'] * std
            h = self.config['cusum_h_factor'] * std
            
            cusum_pos = 0
            cusum_neg = 0
            
            for i in range(1, len(history)):
                diff = history[i] - history[i-1]
                cusum_pos = max(0, cusum_pos + diff - k)
                cusum_neg = max(0, cusum_neg - diff - k)
                
                if cusum_pos > h or cusum_neg > h:
                    reason = f"CUSUM change detected (pos: {cusum_pos:.2f}, neg: {cusum_neg:.2f})"
                    metadata['reset_type'] = 'cusum'
                    metadata['cusum_pos'] = cusum_pos
                    metadata['cusum_neg'] = cusum_neg
                    state['measurement_history'] = [current_weight]
                    return True, reason, metadata
        
        return False, "No change point detected", metadata
    
    def _combined_reset(
        self,
        current_weight: float,
        timestamp: datetime,
        source: str,
        state: Dict,
        gap_days: float,
        metadata: Dict
    ) -> Tuple[bool, str, Dict]:
        """Combine multiple methods with voting."""
        
        votes = []
        reasons = []
        vote_details = {}
        
        if self.config['enable_questionnaire_gap']:
            reset, reason, _ = self._questionnaire_gap_reset(source, gap_days, state, {})
            if reset:
                votes.append('questionnaire')
                reasons.append(reason)
                vote_details['questionnaire'] = reason
        
        if self.config['enable_variance_reset']:
            reset, reason, _ = self._variance_reset(current_weight, state, {})
            if reset:
                votes.append('variance')
                reasons.append(reason)
                vote_details['variance'] = reason
        
        if self.config['enable_reliability_reset']:
            reset, reason, _ = self._reliability_reset(source, gap_days, {})
            if reset:
                votes.append('reliability')
                reasons.append(reason)
                vote_details['reliability'] = reason
        
        if self.config['enable_changepoint_reset']:
            reset, reason, _ = self._changepoint_reset(current_weight, state, {})
            if reset:
                votes.append('changepoint')
                reasons.append(reason)
                vote_details['changepoint'] = reason
        
        metadata['votes'] = votes
        metadata['vote_count'] = len(votes)
        metadata['vote_details'] = vote_details
        metadata['vote_threshold'] = self.config['combined_vote_threshold']
        
        if len(votes) >= self.config['combined_vote_threshold']:
            reason = f"Combined reset ({len(votes)} triggers: {', '.join(votes)})"
            metadata['reset_type'] = 'combined'
            return True, reason, metadata
        
        return False, f"Insufficient votes ({len(votes)}/{self.config['combined_vote_threshold']})", metadata
    
    def track_reset(
        self,
        timestamp: datetime,
        source: str,
        reason: str,
        metadata: Dict
    ):
        """
        Track reset event for analysis.
        
        Args:
            timestamp: Reset timestamp
            source: Data source that triggered reset
            reason: Reset reason
            metadata: Additional metadata
        """
        self.reset_history.append({
            'timestamp': timestamp,
            'source': source,
            'reason': reason,
            'metadata': metadata
        })
    
    def get_reset_statistics(self) -> Dict:
        """
        Get statistics about recent resets.
        
        Returns:
            Dictionary with reset statistics
        """
        if not self.reset_history:
            return {
                'total_resets': 0,
                'reset_types': {},
                'average_gap_days': 0
            }
        
        reset_types = {}
        total_gap_days = 0
        
        for reset in self.reset_history:
            reset_type = reset.get('metadata', {}).get('reset_type', 'unknown')
            reset_types[reset_type] = reset_types.get(reset_type, 0) + 1
            total_gap_days += reset.get('metadata', {}).get('gap_days', 0)
        
        return {
            'total_resets': len(self.reset_history),
            'reset_types': reset_types,
            'average_gap_days': total_gap_days / len(self.reset_history) if self.reset_history else 0,
            'recent_resets': list(self.reset_history)[-5:]
        }
    
    def update_state_with_source(self, state: Dict, source: str):
        """
        Update state with source information for next reset decision.
        
        Args:
            state: Processor state to update
            source: Current source identifier
        """
        state['last_source'] = source
        
        if self.config['enable_changepoint_reset'] and 'measurement_history' not in state:
            state['measurement_history'] = []