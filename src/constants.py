"""
Constants for weight stream processor.
All hard-coded values that should not be configurable for safety.
"""

from dataclasses import dataclass
from typing import Dict, Optional


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

# Physiological limits (hard-coded for safety)
PHYSIOLOGICAL_LIMITS = {
    'ABSOLUTE_MIN_WEIGHT': 30.0,  # kg
    'ABSOLUTE_MAX_WEIGHT': 400.0,  # kg
    'SUSPICIOUS_MIN_WEIGHT': 40.0,  # kg
    'SUSPICIOUS_MAX_WEIGHT': 300.0,  # kg
    'DEFAULT_HEIGHT_M': 1.67,
    'MAX_DAILY_CHANGE_KG': 6.44,  # Optimized from 5.0
    'MAX_WEEKLY_CHANGE_KG': 10.0,  # kg
    'TYPICAL_DAILY_VARIATION_KG': 2.0,  # kg
    'MAX_SUSTAINED_DAILY_KG': 2.57,  # Optimized from 1.5
    'MAX_CHANGE_1H': 4.22,  # New optimized value
    'MAX_CHANGE_6H': 6.23,  # New optimized value
    'MAX_CHANGE_24H': 6.44,  # New optimized value
    'LIMIT_TOLERANCE': 0.2493,  # Optimized from 0.10
    'SUSTAINED_TOLERANCE': 0.50,  # Optimized from 0.25
    'SESSION_VARIANCE': 5.81  # Optimized from 5.0
}

# BMI limits
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

# Kalman filter defaults
KALMAN_DEFAULTS = {
    'initial_variance': 0.361,  # Optimized from 1.0
    'transition_covariance_weight': 0.0160,  # Optimized from 0.1
    'transition_covariance_trend': 0.0001,  # Optimized from 0.001
    'observation_covariance': 3.490,  # Optimized from 1.0
    'reset_gap_days': 30,
    'questionnaire_reset_days': 10
}

# Source profiles with reliability and noise characteristics
SOURCE_PROFILES = {
    'care-team-upload': {
        'outlier_rate': 3.6,
        'reliability': 'excellent',
        'noise_multiplier': 0.5,
        'priority': 1,
        'base_threshold_kg': 2.0,
        'max_threshold_kg': 8.0,
        'spike_threshold_kg': 5.0,
        'drift_threshold_kg': 3.0,
        'noise_threshold_kg': 2.0,
        'max_daily_change_kg': 6.44,
        'high_confidence_multiplier': 0.8,
        'low_confidence_multiplier': 1.5
    },
    'patient-upload': {
        'outlier_rate': 13.0,
        'reliability': 'excellent',
        'noise_multiplier': 0.7,
        'priority': 4,
        'base_threshold_kg': 2.0,
        'max_threshold_kg': 10.0,
        'spike_threshold_kg': 5.0,
        'drift_threshold_kg': 3.0,
        'noise_threshold_kg': 2.0,
        'max_daily_change_kg': 6.44,
        'high_confidence_multiplier': 0.8,
        'low_confidence_multiplier': 1.5
    },
    'internal-questionnaire': {
        'outlier_rate': 14.0,
        'reliability': 'good',
        'noise_multiplier': 0.8,
        'priority': 1,
        'base_threshold_kg': 3.0,
        'max_threshold_kg': 12.0,
        'spike_threshold_kg': 6.0,
        'drift_threshold_kg': 4.0,
        'noise_threshold_kg': 3.0,
        'max_daily_change_kg': 8.0,
        'high_confidence_multiplier': 0.9,
        'low_confidence_multiplier': 1.8
    },
    'initial-questionnaire': {
        'outlier_rate': 14.0,
        'reliability': 'good',
        'noise_multiplier': 0.8,
        'priority': 1,
        'base_threshold_kg': 3.0,
        'max_threshold_kg': 12.0,
        'spike_threshold_kg': 6.0,
        'drift_threshold_kg': 4.0,
        'noise_threshold_kg': 3.0,
        'max_daily_change_kg': 8.0,
        'high_confidence_multiplier': 0.9,
        'low_confidence_multiplier': 1.8
    },
    'patient-device': {
        'outlier_rate': 20.7,
        'reliability': 'good',
        'noise_multiplier': 1.0,
        'priority': 3,
        'base_threshold_kg': 2.5,
        'max_threshold_kg': 10.0,
        'spike_threshold_kg': 5.5,
        'drift_threshold_kg': 3.5,
        'noise_threshold_kg': 2.5,
        'max_daily_change_kg': 7.0,
        'high_confidence_multiplier': 0.85,
        'low_confidence_multiplier': 1.6
    },
    'https://connectivehealth.io': {
        'outlier_rate': 35.8,
        'reliability': 'moderate',
        'noise_multiplier': 1.5,
        'priority': 2,
        'base_threshold_kg': 4.0,
        'max_threshold_kg': 15.0,
        'spike_threshold_kg': 8.0,
        'drift_threshold_kg': 5.0,
        'noise_threshold_kg': 4.0,
        'max_daily_change_kg': 10.0,
        'high_confidence_multiplier': 1.0,
        'low_confidence_multiplier': 2.0
    },
    'https://api.iglucose.com': {
        'outlier_rate': 151.4,
        'reliability': 'poor',
        'noise_multiplier': 3.0,
        'priority': 2,
        'base_threshold_kg': 5.0,
        'max_threshold_kg': 20.0,
        'spike_threshold_kg': 10.0,
        'drift_threshold_kg': 7.0,
        'noise_threshold_kg': 5.0,
        'max_daily_change_kg': 15.0,
        'high_confidence_multiplier': 1.2,
        'low_confidence_multiplier': 2.5
    }
}

DEFAULT_PROFILE = {
    'outlier_rate': 20.0,
    'reliability': 'unknown',
    'noise_multiplier': 1.0,
    'priority': 999,
    'base_threshold_kg': 3.0,
    'max_threshold_kg': 10.0,
    'spike_threshold_kg': 5.0,
    'drift_threshold_kg': 3.0,
    'noise_threshold_kg': 2.0,
    'max_daily_change_kg': 6.44,
    'high_confidence_multiplier': 1.0,
    'low_confidence_multiplier': 1.5
}

# Questionnaire sources (for special handling)
QUESTIONNAIRE_SOURCES = {
    'internal-questionnaire',
    'initial-questionnaire',
    'care-team-upload',
    'questionnaire'
}

# Processing defaults
PROCESSING_DEFAULTS = {
    'extreme_threshold': 0.15,
    'max_daily_change': 0.05,
    'min_weight': 30,
    'max_weight': 400,
    'kalman_cleanup_threshold': 2.0
}

# Quality scoring defaults
QUALITY_SCORING_DEFAULTS = {
    'enabled': False,
    'threshold': 0.6,
    'use_harmonic_mean': True,
    'component_weights': {
        'safety': 0.35,
        'plausibility': 0.25,
        'consistency': 0.25,
        'reliability': 0.15
    },
    'safety_critical_threshold': 0.3,
    'history_window_size': 20
}

# Visualization marker symbols for source types
SOURCE_MARKER_SYMBOLS = {
    'care-team-upload': 'triangle-up',
    'patient-upload': 'circle',
    'internal-questionnaire': 'square',
    'initial-questionnaire': 'square',
    'patient-device': 'diamond',
    'https://connectivehealth.io': 'hexagon',
    'https://api.iglucose.com': 'hexagon',
    'questionnaire': 'square',
    'default': 'circle'
}

# Rejection severity color mapping
REJECTION_SEVERITY_COLORS = {
    'Critical': '#8B0000',  # Dark red for impossible values
    'High': '#CC0000',      # Medium-dark red for extreme deviations
    'Medium': '#FF4444',    # Medium red for suspicious values
    'Low': '#FF9999'        # Light red for minor issues
}

# Session detection
SESSION_TIMEOUT_MINUTES = 5.0
SESSION_VARIANCE_THRESHOLD = 5.81  # kg

# Helper functions that were in models.py

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