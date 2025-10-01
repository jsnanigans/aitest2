"""
Constants for weight stream processor.
Safety limits and immutable values.
Configurable parameters are now in config.toml
"""

from dataclasses import dataclass
from typing import Dict, Optional

# NOTE: SOURCE_PROFILES and KALMAN_DEFAULTS have been moved to config.toml
# Use ConfigManager to load these values dynamically


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
        return {"value": self.value, "unit": self.unit, "metadata": self.metadata}


# Physiological limits
PHYSIOLOGICAL_LIMITS = {
    "ABSOLUTE_MIN_WEIGHT": 30.0,  # kg
    "ABSOLUTE_MAX_WEIGHT": 400.0,  # kg
    "SUSPICIOUS_MIN_WEIGHT": 40.0,  # kg
    "SUSPICIOUS_MAX_WEIGHT": 300.0,  # kg
    "DEFAULT_HEIGHT_M": 1.67,
    "MAX_DAILY_CHANGE_KG": 2.0,  # Realistic daily fluctuation
    "MAX_WEEKLY_CHANGE_KG": 3.5,  # Aggressive but possible weight loss/gain
    "TYPICAL_DAILY_VARIATION_KG": 1.5,  # Normal daily variation
    "MAX_SUSTAINED_DAILY_KG": 0.5,  # Sustainable long-term change
    "MAX_CHANGE_1H": 1.0,  # Water/food intake immediate effect
    "MAX_CHANGE_6H": 3.0,  # Half-day variation (increased for meal+water+exercise)
    "MAX_CHANGE_24H": 4.0,  # Full day variation (meal cycles + exercise + hydration)
    "MAX_CHANGE_1MIN": 0.5,  # Scale variance + positioning tolerance
    "MAX_CHANGE_5MIN": 1.0,  # Water/bathroom + multiple measurements
    "MAX_MONTHLY_PERCENT": 15,  # Maximum 15% body weight change per month
    "LIMIT_TOLERANCE": 0.1,  # Optimized from 0.10
    "SUSTAINED_TOLERANCE": 0.25,  # Optimized from 0.25
    "SESSION_VARIANCE": 2,  # Optimized from 5.0
}

# Supported weight units - STRICT WHITELIST
SUPPORTED_WEIGHT_UNITS = {
    # Metric units
    "kg",
    "kilogram",
    "kilograms",
    "g",
    "gram",
    "grams",
    # Imperial units
    "lb",
    "lbs",
    "pound",
    "pounds",
    "st",
    "stone",
    "stones",
}

# BMI limits
BMI_LIMITS = {
    "CRITICAL_LOW": 15.0,
    "SEVERE_LOW": 16.0,
    "UNDERWEIGHT": 18.5,
    "OVERWEIGHT": 25.0,
    "OBESE": 30.0,
    "SEVERE_OBESE": 35.0,
    "MORBID_OBESE": 40.0,
    "CRITICAL_HIGH": 50.0,
    "IMPOSSIBLE_LOW": 17.0,
    "IMPOSSIBLE_HIGH": 100.0,
    "SUSPICIOUS_LOW": 20.0,
    "SUSPICIOUS_HIGH": 70.0,
}

# ==============================================================================
# DEPRECATED CONSTANTS - Moved to config.toml
# ==============================================================================
# These remain here for backward compatibility but are now loaded from config
# Use ConfigManager.load_config() and access config["sources"] instead

# Module-level variables that get populated from config on first access
_SOURCE_PROFILES: Optional[Dict] = None
_DEFAULT_PROFILE: Optional[Dict] = None
_PROFILES_LOADED = False


def _ensure_profiles_loaded():
    """Ensure source profiles are loaded from config."""
    global _SOURCE_PROFILES, _DEFAULT_PROFILE, _PROFILES_LOADED
    if not _PROFILES_LOADED:
        try:
            # Lazy import to avoid circular dependency
            from ..aws.config.config_manager import ConfigManager
            config = ConfigManager.load_config()
            _SOURCE_PROFILES = config.get("sources", {})
            _DEFAULT_PROFILE = _SOURCE_PROFILES.get("default", {
                "outlier_rate": 20.0,
                "reliability": "unknown",
                "noise_multiplier": 1.0,
                "priority": 999,
                "base_threshold_kg": 3.0,
                "max_threshold_kg": 10.0,
            })
        except Exception as e:
            # Fallback if config can't be loaded
            _SOURCE_PROFILES = {}
            _DEFAULT_PROFILE = {
                "noise_multiplier": 1.0,
                "priority": 999,
                "reliability": "unknown",
            }
        _PROFILES_LOADED = True


# Lazy load on module import
_ensure_profiles_loaded()

# Expose as module-level constants for backward compatibility
SOURCE_PROFILES = _SOURCE_PROFILES
DEFAULT_PROFILE = _DEFAULT_PROFILE

# Questionnaire sources (for special handling)
QUESTIONNAIRE_SOURCES = {
    "internal-questionnaire",
    "initial-questionnaire",
    "care-team-upload",
    "questionnaire",
}

# DEPRECATED: Kalman defaults now in config.toml
KALMAN_DEFAULTS = {
    "initial_variance": 0.364,
    "transition_covariance_weight": 0.018,
    "transition_covariance_trend": 0.00015,
    "observation_covariance": 3.4,
}

# Visualization marker symbols for source types
SOURCE_MARKER_SYMBOLS = {
    "care-team-upload": "triangle-up",
    "patient-upload": "circle",
    "internal-questionnaire": "square",
    "initial-questionnaire": "square",
    "patient-device": "diamond",
    "https://connectivehealth.io": "hexagon",
    "https://api.iglucose.com": "hexagon",
    "questionnaire": "square",
    "default": "circle",
}

# Rejection severity color mapping
REJECTION_SEVERITY_COLORS = {
    "Critical": "#8B0000",  # Dark red for impossible values
    "High": "#CC0000",  # Medium-dark red for extreme deviations
    "Medium": "#FF4444",  # Medium red for suspicious values
    "Low": "#FF9999",  # Light red for minor issues
}

# Session detection
SESSION_TIMEOUT_MINUTES = 5.0
SESSION_VARIANCE_THRESHOLD = 5.81  # kg

# Helper functions that were in models.py


def get_source_priority(source: str) -> int:
    """Get priority for a source (lower number = higher priority)."""
    _ensure_profiles_loaded()
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile.get("priority", 999)


def get_source_reliability(source: str) -> str:
    """Get reliability classification for source."""
    _ensure_profiles_loaded()
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile.get("reliability", "unknown")


def get_noise_multiplier(source: str) -> float:
    """Get Kalman filter measurement noise multiplier for source."""
    _ensure_profiles_loaded()
    profile = SOURCE_PROFILES.get(source, DEFAULT_PROFILE)
    return profile.get("noise_multiplier", 1.0)


def categorize_rejection_enhanced(reason: str) -> str:
    """Enhanced categorization including BMI and unit issues."""
    reason_lower = reason.lower()

    if "bmi" in reason_lower:
        return "BMI_Detection"
    elif (
        "unit" in reason_lower
        or "pound" in reason_lower
        or "conversion" in reason_lower
    ):
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
