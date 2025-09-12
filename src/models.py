"""
Data models, constants, and helper functions for weight processing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from collections import defaultdict


@dataclass
class ThresholdResult:
    """Result from threshold calculation with explicit units."""
    
    value: float
    unit: str
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'value': self.value,
            'unit': self.unit,
            'metadata': self.metadata
        }


SOURCE_PROFILES = {
    'care-team-upload': {
        'outlier_rate': 3.6,
        'reliability': 'excellent',
        'noise_multiplier': 0.5,
        'priority': 1
    },
    'patient-upload': {
        'outlier_rate': 13.0,
        'reliability': 'excellent',
        'noise_multiplier': 0.7,
        'priority': 4
    },
    'internal-questionnaire': {
        'outlier_rate': 14.0,
        'reliability': 'good',
        'noise_multiplier': 0.8,
        'priority': 1
    },
    'initial-questionnaire': {
        'outlier_rate': 14.0,
        'reliability': 'good',
        'noise_multiplier': 0.8,
        'priority': 1
    },
    'patient-device': {
        'outlier_rate': 20.7,
        'reliability': 'good',
        'noise_multiplier': 1.0,
        'priority': 3
    },
    'https://connectivehealth.io': {
        'outlier_rate': 35.8,
        'reliability': 'moderate',
        'noise_multiplier': 1.5,
        'priority': 2
    },
    'https://api.iglucose.com': {
        'outlier_rate': 151.4,
        'reliability': 'poor',
        'noise_multiplier': 3.0,
        'priority': 2
    }
}

DEFAULT_PROFILE = {
    'outlier_rate': 20.0,
    'reliability': 'unknown',
    'noise_multiplier': 1.0,
    'priority': 999
}

QUESTIONNAIRE_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'care-team-upload',
    'questionnaire'
}

BMI_LIMITS = {
    'CRITICAL_LOW': 15.0,
    'SEVERE_LOW': 16.0,
    'UNDERWEIGHT': 18.5,
    'OVERWEIGHT': 25.0,
    'OBESE': 30.0,
    'SEVERE_OBESE': 35.0,
    'MORBID_OBESE': 40.0,
    'CRITICAL_HIGH': 50.0,
    'IMPOSSIBLE_LOW': 10.0,
    'IMPOSSIBLE_HIGH': 100.0,
    'SUSPICIOUS_LOW': 13.0,
    'SUSPICIOUS_HIGH': 60.0
}

PHYSIOLOGICAL_LIMITS = {
    'MIN_WEIGHT': 30,
    'MAX_WEIGHT': 400,
    'DEFAULT_HEIGHT_M': 1.67,
    'MAX_DAILY_CHANGE_KG': 5.0,
    'MAX_SUSTAINED_DAILY_KG': 1.5
}

KALMAN_DEFAULTS = {
    'initial_variance': 1.0,
    'transition_covariance_weight': 0.1,
    'transition_covariance_trend': 0.001,
    'observation_covariance': 1.0,
    'reset_gap_days': 30,
    'questionnaire_reset_days': 10
}

PROCESSING_DEFAULTS = {
    'extreme_threshold': 0.15,
    'max_daily_change': 0.05,
    'min_weight': 30,
    'max_weight': 400,
    'kalman_cleanup_threshold': 2.0
}


def categorize_rejection_enhanced(reason: str) -> str:
    """Enhanced categorization including BMI and unit issues."""
    reason_lower = reason.lower()
    
    if "bmi" in reason_lower:
        return "BMI_Detection"
    elif "unit" in reason_lower or "pound" in reason_lower or "conversion" in reason_lower:
        return "Unit_Conversion"
    elif "physiological" in reason_lower:
        return "Physiological_Limit"
    elif "outside bounds" in reason_lower:
        return "Bounds"
    elif "extreme deviation" in reason_lower:
        return "Extreme"
    elif "session variance" in reason_lower or "different user" in reason_lower:
        return "Variance"
    elif "sustained" in reason_lower:
        return "Sustained"
    elif "daily fluctuation" in reason_lower:
        return "Daily"
    else:
        return "Other"


def get_rejection_severity(reason: str, weight_change: float = 0) -> str:
    """Determine severity of rejection."""
    reason_lower = reason.lower()
    
    if "impossible" in reason_lower or "physiologically impossible" in reason_lower:
        return "Critical"
    elif "extreme" in reason_lower or weight_change > 20:
        return "High"
    elif "suspicious" in reason_lower or weight_change > 10:
        return "Medium"
    else:
        return "Low"


def get_source_priority(source: str) -> int:
    """Get priority for a source (lower number = higher priority)."""
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile.get('priority', 999)


def get_source_reliability(source: str) -> str:
    """Get reliability classification for source."""
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile['reliability']


def get_noise_multiplier(source: str) -> float:
    """Get Kalman filter measurement noise multiplier for source."""
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile['noise_multiplier']