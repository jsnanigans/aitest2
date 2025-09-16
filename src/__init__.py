"""
Weight Stream Processor Package
"""

# Core processing
from .processing.processor import (
    WeightProcessor,
    process_weight_enhanced,
    process_measurement
)

# Database
from .database.database import (
    ProcessorStateDB,
    ProcessorDatabase,
    get_state_db
)

# Visualization - unified module
from .viz.visualization import create_weight_timeline

# Constants
from .constants import (
    ThresholdResult,
    SOURCE_PROFILES,
    DEFAULT_PROFILE,
    BMI_LIMITS,
    PHYSIOLOGICAL_LIMITS,
    QUESTIONNAIRE_SOURCES,
    KALMAN_DEFAULTS,
    PROCESSING_DEFAULTS,
    categorize_rejection_enhanced,
    get_rejection_severity,
    get_source_priority,
    get_source_reliability,
    get_noise_multiplier
)

# Validation
from .processing.validation import (
    BMIValidator,
    ThresholdCalculator,
    PhysiologicalValidator,
    DataQualityPreprocessor
)

# Kalman filter
from .processing.kalman import KalmanFilterManager

# Quality scoring
from .processing.quality_scorer import (
    QualityScorer,
    QualityScore,
    MeasurementHistory
)

# Utilities
from .utils import (
    StructuredLogger,
    PerformanceTimer,
    VizLogger,
    get_logger,
    set_verbosity,
    format_timestamp,
    safe_divide
)

__all__ = [
    # Core processor
    'WeightProcessor',
    'process_weight_enhanced',
    'process_measurement',
    
    # Database
    'ProcessorStateDB',
    'ProcessorDatabase',
    'get_state_db',
    
    # Visualization
    'create_weight_timeline',
    
    # Constants
    'ThresholdResult',
    'SOURCE_PROFILES',
    'DEFAULT_PROFILE',
    'BMI_LIMITS',
    'PHYSIOLOGICAL_LIMITS',
    'QUESTIONNAIRE_SOURCES',
    'KALMAN_DEFAULTS',
    'PROCESSING_DEFAULTS',
    'categorize_rejection_enhanced',
    'get_rejection_severity',
    'get_source_priority',
    'get_source_reliability',
    'get_noise_multiplier',
    
    # Validation
    'BMIValidator',
    'ThresholdCalculator',
    'PhysiologicalValidator',
    'DataQualityPreprocessor',
    
    # Kalman
    'KalmanFilterManager',
    
    # Quality
    'QualityScorer',
    'QualityScore',
    'MeasurementHistory',
    
    # Utilities
    'StructuredLogger',
    'PerformanceTimer',
    'VizLogger',
    'get_logger',
    'set_verbosity',
    'format_timestamp',
    'safe_divide',
]