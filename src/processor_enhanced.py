"""
Enhanced weight processor with data quality improvements.
Based on analysis of 709,246 measurements revealing source-specific patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque
import numpy as np

from processor import WeightProcessor
from processor_database import get_state_db


class DataQualityPreprocessor:
    """Pre-process and clean data before Kalman filtering."""
    
    # Sources with high pound usage (>70%)
    POUND_SOURCES = {
        'patient-upload',
        'internal-questionnaire', 
        'care-team-upload'
    }
    
    @staticmethod
    def preprocess(weight: float, source: str, timestamp: datetime) -> Tuple[Optional[float], Dict]:
        """
        Clean and standardize weight data.
        
        Returns:
            (cleaned_weight, metadata) or (None, metadata) if rejected
        """
        metadata = {
            'original_weight': weight,
            'source': source,
            'timestamp': timestamp.isoformat(),
            'corrections': [],
            'warnings': []
        }
        
        # 1. Detect and handle BMI values (15-50 range)
        if 15 <= weight <= 50:
            # Check if source is known to send BMI
            if 'connectivehealth' in source.lower():
                metadata['warnings'].append('Possible BMI value detected')
                # Could attempt to reverse-calculate if we had height
                # For now, flag for review
                if weight < 30:  # Very likely BMI
                    metadata['rejected'] = 'Suspected BMI value, not weight'
                    return None, metadata
        
        # 2. Handle pound entries for sources known to use pounds
        if source in DataQualityPreprocessor.POUND_SOURCES:
            # Check if value is in typical pound range
            if 80 <= weight <= 450:  # Typical pound range for adults
                # High confidence this is pounds
                weight_kg = weight * 0.453592
                metadata['corrections'].append(f'Converted {weight:.1f} lb to {weight_kg:.1f} kg')
                weight = weight_kg
        
        # 3. Handle extreme outliers that are clearly errors
        if weight < 20:
            metadata['warnings'].append('Weight below 20kg - possible unit error')
            # Might be in stones
            if 5 <= weight <= 30:
                weight_kg = weight * 6.35029
                metadata['corrections'].append(f'Converted {weight:.1f} st to {weight_kg:.1f} kg')
                weight = weight_kg
        
        if weight > 500:
            metadata['rejected'] = 'Weight above 500kg - data error'
            return None, metadata
        
        # 4. Flag high-risk sources
        if 'iglucose' in source.lower():
            metadata['warnings'].append('High-outlier source - increased scrutiny')
            metadata['high_risk'] = True
        
        return weight, metadata


class AdaptiveOutlierDetector:
    """Adaptive outlier detection based on source reliability."""
    
    # Outlier rates per 1000 from analysis
    SOURCE_PROFILES = {
        'care-team-upload': {
            'outlier_rate': 3.6,
            'reliability': 'excellent'
        },
        'patient-upload': {
            'outlier_rate': 13.0,
            'reliability': 'excellent'
        },
        'internal-questionnaire': {
            'outlier_rate': 14.0,
            'reliability': 'good'
        },
        'patient-device': {
            'outlier_rate': 20.7,
            'reliability': 'good'
        },
        'https://connectivehealth.io': {
            'outlier_rate': 35.8,
            'reliability': 'moderate'
        },
        'https://api.iglucose.com': {
            'outlier_rate': 151.4,
            'reliability': 'poor'
        }
    }
    
    @staticmethod
    def get_adaptive_threshold(source: str, time_gap_days: int) -> float:
        """
        Get outlier threshold based on source reliability and time gap.
        
        Args:
            source: Data source identifier
            time_gap_days: Days since last measurement
            
        Returns:
            Maximum acceptable weight change in kg
        """
        profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
        outlier_rate = profile.get('outlier_rate', 50.0)
        
        # Base physiological limit: 2kg/week is extreme but possible
        physiological_rate = 2.0 / 7.0  # kg per day
        
        if outlier_rate > 100:  # Very unreliable (iGlucose)
            # Much stricter - only allow 1kg/week
            max_rate = 1.0 / 7.0
            threshold = max(3.0, max_rate * time_gap_days)
        elif outlier_rate > 30:  # Moderate reliability
            # Standard physiological limit
            max_rate = physiological_rate
            threshold = max(5.0, max_rate * time_gap_days)
        else:  # Excellent reliability
            # More lenient - trust the source more
            max_rate = physiological_rate * 1.5
            threshold = max(10.0, max_rate * time_gap_days)
        
        # Cap at reasonable maximum
        return min(threshold, 20.0)
    
    @staticmethod
    def check_outlier(
        weight_change: float,
        source: str,
        time_gap_days: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if weight change is an outlier.
        
        Returns:
            (is_outlier, reason)
        """
        threshold = AdaptiveOutlierDetector.get_adaptive_threshold(source, time_gap_days)
        
        if abs(weight_change) > threshold:
            profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
            reliability = profile.get('reliability', 'unknown')
            
            reason = (f"Weight change {weight_change:.1f}kg exceeds threshold "
                     f"{threshold:.1f}kg for {reliability} source over {time_gap_days} days")
            return True, reason
        
        return False, None


class AdaptiveKalmanConfig:
    """Adapt Kalman filter parameters based on source quality."""
    
    # Noise multipliers based on source reliability
    NOISE_MULTIPLIERS = {
        'care-team-upload': 0.5,           # Most reliable - trust more
        'patient-upload': 0.7,              # Very reliable
        'internal-questionnaire': 0.8,      # Reliable but sparse
        'patient-device': 1.0,              # Baseline
        'https://connectivehealth.io': 1.5, # Less reliable
        'https://api.iglucose.com': 3.0     # Least reliable - trust less
    }
    
    @staticmethod
    def get_adapted_config(source: str, base_config: Dict) -> Dict:
        """
        Adapt Kalman configuration based on source reliability.
        
        Args:
            source: Data source identifier
            base_config: Base Kalman configuration
            
        Returns:
            Adapted configuration
        """
        config = base_config.copy()
        
        # Get noise multiplier for this source
        multiplier = AdaptiveKalmanConfig.NOISE_MULTIPLIERS.get(source, 1.0)
        
        # Adjust measurement noise
        # Higher noise = Kalman trusts measurement less
        base_noise = config.get('measurement_noise', 1.0)
        config['measurement_noise'] = base_noise * multiplier
        
        # For very unreliable sources, also increase initial uncertainty
        if multiplier > 2.0:
            config['initial_uncertainty'] = config.get('initial_uncertainty', 10.0) * 1.5
        
        return config


class SourceQualityMonitor:
    """Monitor source quality in real-time."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize monitor with rolling window."""
        self.window_size = window_size
        self.stats = defaultdict(lambda: {
            'measurements': deque(maxlen=window_size),
            'outliers': deque(maxlen=window_size),
            'rejections': deque(maxlen=window_size),
            'alerts': []
        })
    
    def record_measurement(
        self,
        source: str,
        is_outlier: bool,
        is_rejected: bool
    ) -> Optional[Dict]:
        """
        Record measurement and check for quality issues.
        
        Returns:
            Alert dictionary if quality issue detected, None otherwise
        """
        stats = self.stats[source]
        stats['measurements'].append(1)
        stats['outliers'].append(1 if is_outlier else 0)
        stats['rejections'].append(1 if is_rejected else 0)
        
        # Need minimum measurements for statistics
        if len(stats['measurements']) < 100:
            return None
        
        # Calculate current rates
        outlier_rate = sum(stats['outliers']) / len(stats['measurements']) * 1000
        rejection_rate = sum(stats['rejections']) / len(stats['measurements']) * 1000
        
        # Get expected rates
        profile = AdaptiveOutlierDetector.SOURCE_PROFILES.get(source, {})
        expected_outlier_rate = profile.get('outlier_rate', 30.0)
        
        alert = None
        
        # Check for degradation
        if outlier_rate > expected_outlier_rate * 1.5:
            alert = {
                'type': 'quality_degradation',
                'source': source,
                'metric': 'outlier_rate',
                'current': outlier_rate,
                'expected': expected_outlier_rate,
                'severity': 'high' if outlier_rate > expected_outlier_rate * 2 else 'medium',
                'message': f"{source} outlier rate {outlier_rate:.1f}/1000 "
                          f"exceeds expected {expected_outlier_rate:.1f}/1000"
            }
            stats['alerts'].append(alert)
        
        # Check for improvement in bad sources
        elif source == 'https://api.iglucose.com' and outlier_rate < 100:
            alert = {
                'type': 'quality_improvement',
                'source': source,
                'metric': 'outlier_rate',
                'current': outlier_rate,
                'expected': expected_outlier_rate,
                'severity': 'info',
                'message': f"{source} showing improvement: {outlier_rate:.1f}/1000"
            }
        
        return alert
    
    def get_source_summary(self, source: str) -> Dict:
        """Get quality summary for a source."""
        stats = self.stats[source]
        
        if not stats['measurements']:
            return {'status': 'no_data'}
        
        total = len(stats['measurements'])
        outlier_rate = sum(stats['outliers']) / total * 1000 if total > 0 else 0
        rejection_rate = sum(stats['rejections']) / total * 1000 if total > 0 else 0
        
        return {
            'measurements': total,
            'outlier_rate': outlier_rate,
            'rejection_rate': rejection_rate,
            'recent_alerts': stats['alerts'][-5:] if stats['alerts'] else []
        }


# Global monitor instance
quality_monitor = SourceQualityMonitor()


def process_weight_enhanced(
    user_id: str,
    weight: float,
    timestamp: datetime,
    source: str,
    processing_config: Dict,
    kalman_config: Dict
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
        weight: Weight measurement (may be in pounds)
        timestamp: Measurement timestamp
        source: Data source identifier
        processing_config: Processing configuration
        kalman_config: Kalman filter configuration
        
    Returns:
        Processing result with additional metadata, or None if rejected
    """
    
    # Step 1: Pre-process data
    cleaned_weight, preprocess_metadata = DataQualityPreprocessor.preprocess(
        weight, source, timestamp
    )
    
    if cleaned_weight is None:
        # Rejected in pre-processing
        quality_monitor.record_measurement(source, False, True)
        return {
            'rejected': True,
            'rejection_reason': preprocess_metadata.get('rejected', 'Pre-processing rejection'),
            'preprocessing_metadata': preprocess_metadata
        }
    
    # Step 2: Adapt configuration based on source
    # Get time gap for adaptive thresholds
    db = get_state_db()
    state = db.get_state(user_id)
    
    time_gap_days = 0
    if state and state.get('last_timestamp'):
        time_gap_days = (timestamp - state['last_timestamp']).days
    
    # Adapt outlier threshold
    adapted_config = processing_config.copy()
    adapted_config['extreme_threshold'] = AdaptiveOutlierDetector.get_adaptive_threshold(
        source, time_gap_days
    )
    
    # Step 3: Adapt Kalman configuration
    adapted_kalman = AdaptiveKalmanConfig.get_adapted_config(source, kalman_config)
    
    # Step 4: Process with original Kalman filter
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=cleaned_weight,
        timestamp=timestamp,
        source=source,
        processing_config=adapted_config,
        kalman_config=adapted_kalman
    )
    
    # Step 5: Monitor and enhance result
    if result:
        # Check if it was an outlier
        is_outlier = False
        if state and state.get('last_state') is not None:
            last_weight = state['last_state'][0] if isinstance(state['last_state'], np.ndarray) else state['last_state']
            weight_change = abs(cleaned_weight - last_weight)
            is_outlier, outlier_reason = AdaptiveOutlierDetector.check_outlier(
                weight_change, source, time_gap_days
            )
        
        # Record for monitoring
        is_rejected = result.get('rejected', False)
        alert = quality_monitor.record_measurement(source, is_outlier, is_rejected)
        
        # Enhance result with metadata
        result['preprocessing_metadata'] = preprocess_metadata
        result['adaptive_threshold'] = adapted_config['extreme_threshold']
        result['measurement_noise_used'] = adapted_kalman['measurement_noise']
        
        if alert:
            result['quality_alert'] = alert
        
        # Add source quality summary
        result['source_quality'] = quality_monitor.get_source_summary(source)
    
    return result