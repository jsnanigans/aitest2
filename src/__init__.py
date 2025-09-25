"""
Weight Stream Processor Package
"""

# Core processing
from .core.processing.processor import process_measurement

# Database
from .core.database import get_state_db
from .core.database.base import StateStore

# Constants
from .core.constants import KALMAN_DEFAULTS, PHYSIOLOGICAL_LIMITS, get_noise_multiplier

# Validation
from .core.processing.validation import DataQualityPreprocessor

# Kalman filter
from .core.processing.kalman import KalmanFilterManager

# Quality scoring
from .core.processing.unified_quality_scorer import (
    UnifiedQualityScorer,
    QualityScore,
    MeasurementHistory,
)

# Utilities (if type conversion is needed, import from processing)
# Note: ensure_float is in core.processing.type_conversion if needed

__all__ = [
    # Core processor
    "process_measurement",
    # Database
    "get_state_db",
    "StateStore",
    # Constants
    "KALMAN_DEFAULTS",
    "PHYSIOLOGICAL_LIMITS",
    "get_noise_multiplier",
    # Validation
    "DataQualityPreprocessor",
    # Kalman
    "KalmanFilterManager",
    # Quality
    "UnifiedQualityScorer",
    "QualityScore",
    "MeasurementHistory",
]
