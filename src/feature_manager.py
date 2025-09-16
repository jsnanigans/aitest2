"""
Feature Manager for Weight Processing System
Centralized management of feature toggles with safety guarantees
"""
from typing import Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FeatureManager:
    """Manages feature toggles with dependency checking and safety overrides"""

    # No mandatory features - all can be disabled
    MANDATORY_FEATURES = set()

    # Feature dependencies (feature -> set of required features)
    DEPENDENCIES = {
        'quality_scoring': {'kalman_filtering'},
        'quality_override': {'quality_scoring', 'outlier_detection'},
        'kalman_deviation_check': {'kalman_filtering', 'outlier_detection'},
        'adaptive_parameters': {'kalman_filtering'},
        'reset_tracking': {'state_persistence'},
        'history_buffer': {'state_persistence'},
    }

    # Default feature states (all enabled for backward compatibility)
    DEFAULT_FEATURES = {
        # Core processing
        'kalman_filtering': True,
        'quality_scoring': True,
        'outlier_detection': True,
        'quality_override': True,

        # Validation layers
        'validation': True,
        'physiological_validation': True,  # MANDATORY
        'bmi_checking': True,
        'rate_limiting': True,
        'dangerous_rate_limiting': True,   # MANDATORY
        'temporal_consistency': True,
        'session_variance': True,

        # Outlier detection methods
        'outlier_iqr': True,
        'outlier_mad': True,
        'outlier_temporal': True,
        'kalman_deviation_check': True,

        # Quality scoring components
        'quality_safety': True,
        'quality_plausibility': True,
        'quality_consistency': True,
        'quality_reliability': True,

        # State management
        'state_persistence': True,
        'history_buffer': True,
        'reset_tracking': True,

        # Reset types
        'reset_initial': True,
        'reset_hard': True,
        'reset_soft': True,

        # Adaptive features
        'adaptive_noise': True,
        'adaptive_parameters': True,
        'parameter_decay': True,

        # Processing options
        'retrospective_analysis': True,
        'retrospective_outlier': True,
        'retrospective_rollback': True,

        # Visualization
        'visualization_enabled': True,
        'visualization_threading': True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with config dict containing features section"""
        self.features = self.DEFAULT_FEATURES.copy()
        self.warnings = []

        if config and 'features' in config:
            self._load_features(config['features'])

        self._validate_dependencies()
        self._enforce_mandatory()
        self._log_configuration()

    def _load_features(self, features_config: Dict[str, Any]):
        """Load features from config, handling nested structures"""
        for key, value in features_config.items():
            if isinstance(value, dict):
                # Handle nested sections like features.validation
                for subkey, subvalue in value.items():
                    if key == 'outlier_methods':
                        feature_key = f"outlier_{subkey}"
                    elif key == 'visualization':
                        feature_key = f"visualization_{subkey}"
                    else:
                        feature_key = f"{key}_{subkey}"
                    if feature_key in self.DEFAULT_FEATURES:
                        self.features[feature_key] = bool(subvalue)
            elif isinstance(value, bool):
                # Handle top-level boolean features
                if key in self.DEFAULT_FEATURES:
                    self.features[key] = value

    def _validate_dependencies(self):
        """Check and fix feature dependencies"""
        changes_made = True
        while changes_made:
            changes_made = False
            for feature, deps in self.DEPENDENCIES.items():
                if self.features.get(feature, False):
                    for dep in deps:
                        if not self.features.get(dep, False):
                            logger.warning(
                                f"Feature '{feature}' requires '{dep}'. Enabling '{dep}'."
                            )
                            self.features[dep] = True
                            changes_made = True

    def _enforce_mandatory(self):
        """No mandatory features - all can be disabled"""
        # All features can now be disabled per user request
        pass

    def _log_configuration(self):
        """Log active feature configuration"""
        disabled_features = [k for k, v in self.features.items() if not v]
        if disabled_features:
            logger.info(f"Disabled features: {', '.join(disabled_features)}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(warning)

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.features.get(feature, True)

    def get_enabled_features(self) -> Set[str]:
        """Get set of all enabled features"""
        return {k for k, v in self.features.items() if v}

    def get_disabled_features(self) -> Set[str]:
        """Get set of all disabled features"""
        return {k for k, v in self.features.items() if not v}

    def validate_config(self) -> bool:
        """Validate the current configuration"""
        # Check no mandatory features are disabled
        for feature in self.MANDATORY_FEATURES:
            if not self.features.get(feature, True):
                return False

        # Check all dependencies are met
        for feature, deps in self.DEPENDENCIES.items():
            if self.features.get(feature, False):
                for dep in deps:
                    if not self.features.get(dep, False):
                        return False

        return True

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the feature configuration"""
        return {
            'total_features': len(self.features),
            'enabled': len(self.get_enabled_features()),
            'disabled': len(self.get_disabled_features()),
            'mandatory_enforced': list(self.MANDATORY_FEATURES),
            'warnings': self.warnings,
            'valid': self.validate_config()
        }