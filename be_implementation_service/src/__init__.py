"""
Weight Stream Processor Package - AWS Lambda Implementation

This package provides AWS Lambda-specific code for the weight processor service.
Core processing logic is provided by the weight_processor_lib package.
"""

# Re-export core components from python_lib for convenience
from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.database import get_state_db
from weight_processor_lib.core.database.base import StateStore
from weight_processor_lib.core.constants import (
    KALMAN_DEFAULTS,
    PHYSIOLOGICAL_LIMITS,
    get_noise_multiplier,
)
from weight_processor_lib.core.processing.validation import DataQualityPreprocessor
from weight_processor_lib.core.processing.kalman import KalmanFilterManager
from weight_processor_lib.core.processing.unified_quality_scorer import (
    UnifiedQualityScorer,
    QualityScore,
    MeasurementHistory,
)

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
