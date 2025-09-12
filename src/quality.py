"""
Data quality and preprocessing components for weight processing.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd

try:
    from .models import (
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS,
        get_source_reliability,
        get_noise_multiplier,
        categorize_rejection_enhanced,
        get_rejection_severity
    )
    from .validation import ThresholdCalculator
except ImportError:
    from models import (
        SOURCE_PROFILES,
        DEFAULT_PROFILE,
        BMI_LIMITS,
        PHYSIOLOGICAL_LIMITS,
        get_source_reliability,
        get_noise_multiplier,
        categorize_rejection_enhanced,
        get_rejection_severity
    )
    from validation import ThresholdCalculator


class DataQualityPreprocessor:
    """Pre-process and clean data before Kalman filtering."""
    
    DEFAULT_HEIGHT_M = PHYSIOLOGICAL_LIMITS['DEFAULT_HEIGHT_M']
    
    _height_data = None
    _height_data_loaded = False
    
    @classmethod
    def load_height_data(cls):
        """Load height data from CSV file once."""
        if not cls._height_data_loaded:
            try:
                df = pd.read_csv('data/2025-09-11_height_values_latest.csv')
                df['value_m'] = df.apply(lambda row: cls._convert_height_to_meters(
                    row['value_quantity'], row['unit']
                ), axis=1)
                cls._height_data = df.set_index('user_id')['value_m'].to_dict()
                cls._height_data_loaded = True
                print(f"Loaded height data for {len(cls._height_data)} users")
            except Exception as e:
                print(f"Could not load height data: {e}")
                cls._height_data = {}
                cls._height_data_loaded = True
    
    @staticmethod
    def _convert_height_to_meters(value: float, unit: str) -> float:
        """Convert height to meters."""
        unit_lower = unit.lower() if unit else 'm'
        
        if 'cm' in unit_lower or 'centimeter' in unit_lower:
            return value / 100.0
        elif 'in' in unit_lower or 'inch' in unit_lower:
            return value * 0.0254
        elif 'ft' in unit_lower or 'feet' in unit_lower:
            return value * 0.3048
        elif 'm' in unit_lower or 'meter' in unit_lower:
            return value
        else:
            return value / 100.0
    
    @classmethod
    def get_user_height(cls, user_id: str) -> float:
        """Get user's height in meters, using default if not found."""
        if not cls._height_data_loaded:
            cls.load_height_data()
        
        return cls._height_data.get(user_id, cls.DEFAULT_HEIGHT_M)
    
    @staticmethod
    def preprocess(weight: float, source: str, timestamp: datetime, user_id: Optional[str] = None, unit: str = 'kg') -> Tuple[Optional[float], Dict]:
        """
        Clean and standardize weight data with BMI detection.
        
        Args:
            weight: The weight value
            source: Data source identifier
            timestamp: Measurement timestamp
            user_id: User identifier for height lookup
            unit: Unit of the weight measurement ('kg', 'lb', 'lbs', 'pound', 'pounds', etc.)
        
        Returns:
            (cleaned_weight, metadata) or (None, metadata) if rejected
        """
        metadata = {
            'original_weight': weight,
            'original_unit': unit,
            'source': source,
            'timestamp': timestamp.isoformat(),
            'corrections': [],
            'warnings': [],
            'checks_passed': []
        }
        
        user_height = DataQualityPreprocessor.get_user_height(user_id) if user_id else DataQualityPreprocessor.DEFAULT_HEIGHT_M
        
        unit_lower = unit.lower() if unit else 'kg'
        if unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            weight_kg = weight * 0.453592
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower in ['st', 'stone', 'stones']:
            weight_kg = weight * 6.35029
            metadata['corrections'].append(f'Converted {weight:.1f} {unit} to {weight_kg:.1f} kg')
            weight = weight_kg
        elif unit_lower not in ['kg', 'kilogram', 'kilograms']:
            metadata['warnings'].append(f'Unknown unit: {unit}')
        
        if unit_lower in ['kg', 'kilogram', 'kilograms'] and 15 <= weight <= 50:
            implied_weight = weight * (user_height ** 2)
            
            if 40 <= implied_weight <= 200:
                metadata['warnings'].append(
                    f'Value {weight:.1f} likely BMI (implies {implied_weight:.1f}kg weight for height {user_height:.2f}m)'
                )
                metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                weight = implied_weight
            elif 30 <= implied_weight <= 250:
                if 'connectivehealth' in source.lower():
                    metadata['warnings'].append(f'Value {weight:.1f} appears to be BMI from {source}')
                    metadata['corrections'].append(f'Converted BMI {weight:.1f} to weight {implied_weight:.1f}kg')
                    weight = implied_weight
                else:
                    metadata['warnings'].append(f'Value {weight:.1f} might be BMI')
        
        implied_bmi = weight / (user_height ** 2)
        
        if implied_bmi < BMI_LIMITS['IMPOSSIBLE_LOW']:
            metadata['warnings'].append(
                f'Implied BMI {implied_bmi:.1f} physiologically impossible (height: {user_height:.2f}m)'
            )
            metadata['rejected'] = f'BMI {implied_bmi:.1f} outside physiological limits'
            return None, metadata
        
        if implied_bmi > BMI_LIMITS['IMPOSSIBLE_HIGH']:
            metadata['rejected'] = f'Implied BMI {implied_bmi:.1f} physiologically impossible'
            return None, metadata
        
        if implied_bmi < BMI_LIMITS['SUSPICIOUS_LOW']:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously low')
        
        if implied_bmi > BMI_LIMITS['SUSPICIOUS_HIGH']:
            metadata['warnings'].append(f'Implied BMI {implied_bmi:.1f} suspiciously high')
        
        metadata['checks_passed'].append('physiological_limits')
        metadata['implied_bmi'] = round(implied_bmi, 1)
        metadata['user_height_m'] = round(user_height, 2)
        
        if implied_bmi < BMI_LIMITS['UNDERWEIGHT']:
            metadata['bmi_category'] = 'underweight'
        elif implied_bmi < BMI_LIMITS['OVERWEIGHT']:
            metadata['bmi_category'] = 'normal'
        elif implied_bmi < BMI_LIMITS['OBESE']:
            metadata['bmi_category'] = 'overweight'
        else:
            metadata['bmi_category'] = 'obese'
        
        if 'iglucose' in source.lower():
            metadata['warnings'].append('High-outlier source - increased scrutiny')
            metadata['high_risk'] = True
        
        return weight, metadata


class AdaptiveOutlierDetector:
    """Adaptive outlier detection based on source reliability."""
    
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
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        outlier_rate = profile.get('outlier_rate', 50.0)
        
        physiological_rate = 2.0 / 7.0
        
        if outlier_rate > 100:
            max_rate = 1.0 / 7.0
            threshold = max(3.0, max_rate * time_gap_days)
        elif outlier_rate > 30:
            max_rate = physiological_rate
            threshold = max(5.0, max_rate * time_gap_days)
        else:
            max_rate = physiological_rate * 1.5
            threshold = max(10.0, max_rate * time_gap_days)
        
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
            profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
            reliability = profile.get('reliability', 'unknown')
            
            reason = (f"Weight change {weight_change:.1f}kg exceeds threshold "
                     f"{threshold:.1f}kg for {reliability} source over {time_gap_days} days")
            return True, reason
        
        return False, None


class AdaptiveKalmanConfig:
    """Adapt Kalman filter parameters based on source quality."""
    
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
        
        multiplier = get_noise_multiplier(source)
        
        base_noise = config.get('measurement_noise', 1.0)
        config['measurement_noise'] = base_noise * multiplier
        
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
        
        if len(stats['measurements']) < 100:
            return None
        
        outlier_rate = sum(stats['outliers']) / len(stats['measurements']) * 1000
        rejection_rate = sum(stats['rejections']) / len(stats['measurements']) * 1000
        
        profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
        expected_outlier_rate = profile.get('outlier_rate', 30.0)
        
        alert = None
        
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


quality_monitor = SourceQualityMonitor()