"""
Weight Stream Processor Package
"""

# Import all public APIs for backward compatibility
from .processor import (
    WeightProcessor,
    process_weight_enhanced
)

from .database import (
    ProcessorStateDB,
    ProcessorDatabase,
    get_state_db
)



from .visualization import create_dashboard

from .constants import (
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
    PhysiologicalValidator,
    DataQualityPreprocessor
)

from .kalman import KalmanFilterManager

__all__ = [
    # Core processor
    'WeightProcessor',
    'process_weight_enhanced',
    
    # Database
    'ProcessorStateDB',
    'ProcessorDatabase',
    'get_state_db',
    

    
    # Visualization
    'create_dashboard',
    
    # Validation
    'BMIValidator',
    'ThresholdCalculator',
    'PhysiologicalValidator',
    'DataQualityPreprocessor',
    
    # Kalman
    'KalmanFilterManager',
]