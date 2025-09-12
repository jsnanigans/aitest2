"""
Weight Stream Processor Package
"""

# Import all public APIs for backward compatibility
from .processor import (
    WeightProcessor,
    process_weight_enhanced,
    DynamicResetManager,
    categorize_rejection_enhanced,
    get_rejection_severity
)

from .database import (
    ProcessorStateDB,
    ProcessorDatabase,
    get_state_db
)

from .reprocessor import WeightReprocessor

from .visualization import create_dashboard

from .models import (
    ThresholdResult,
    SOURCE_PROFILES,
    DEFAULT_PROFILE,
    BMI_LIMITS,
    PHYSIOLOGICAL_LIMITS,
    QUESTIONNAIRE_SOURCES,
    KALMAN_DEFAULTS,
    PROCESSING_DEFAULTS
)

from .validation import (
    BMIValidator,
    ThresholdCalculator,
    PhysiologicalValidator
)

from .quality import (
    DataQualityPreprocessor,
    AdaptiveOutlierDetector,
    AdaptiveKalmanConfig,
    SourceQualityMonitor,
    quality_monitor
)

from .kalman import KalmanFilterManager

__all__ = [
    # Core processor
    'WeightProcessor',
    'process_weight_enhanced',
    'DynamicResetManager',
    
    # Database
    'ProcessorStateDB',
    'ProcessorDatabase',
    'get_state_db',
    
    # Reprocessor
    'WeightReprocessor',
    
    # Visualization
    'create_dashboard',
    
    # Models
    'ThresholdResult',
    
    # Validation
    'BMIValidator',
    'ThresholdCalculator',
    'PhysiologicalValidator',
    
    # Quality
    'DataQualityPreprocessor',
    'AdaptiveOutlierDetector',
    'AdaptiveKalmanConfig',
    'SourceQualityMonitor',
    'quality_monitor',
    
    # Kalman
    'KalmanFilterManager',
    
    # Helper functions
    'categorize_rejection_enhanced',
    'get_rejection_severity',
]