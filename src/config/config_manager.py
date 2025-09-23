"""Configuration management for multiple environments."""

import os
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration from multiple sources."""

    _cached_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(cls, source: str = 'auto', config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment.

        Args:
            source: 'file', 'env', or 'auto'
            config_path: Path to config file (for 'file' source)

        Returns:
            Configuration dictionary
        """
        # Use cached config if available
        if cls._cached_config is not None:
            return cls._cached_config

        # Determine source
        if source == 'auto':
            if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
                source = 'env'
            else:
                source = 'file'

        # Load configuration
        if source == 'env':
            config = cls._load_from_env()
        else:
            config = cls._load_from_file(config_path or 'config.toml')

        # Cache the config
        cls._cached_config = config
        return config

    @classmethod
    def _load_from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'data': {
                'max_users': int(os.getenv('MAX_USERS', '0')),
                'min_readings': int(os.getenv('MIN_READINGS', '0'))
            },
            'kalman': {
                'enabled': os.getenv('KALMAN_ENABLED', 'true').lower() == 'true',
                'adaptive': os.getenv('KALMAN_ADAPTIVE', 'true').lower() == 'true',
                'process_noise': float(os.getenv('KALMAN_PROCESS_NOISE', '1.0')),
                'observation_noise': float(os.getenv('KALMAN_OBS_NOISE', '4.0')),
                'adaptation': {
                    'enabled': os.getenv('KALMAN_ADAPTATION_ENABLED', 'true').lower() == 'true',
                    'initial_multiplier': float(os.getenv('KALMAN_ADAPTATION_MULTIPLIER', '10.0')),
                    'decay_rate': float(os.getenv('KALMAN_ADAPTATION_DECAY', '0.1'))
                },
                'resets': {
                    'hard_gap_days': int(os.getenv('KALMAN_HARD_GAP_DAYS', '30')),
                    'soft_sources': os.getenv('KALMAN_SOFT_SOURCES', 'questionnaire').split(',')
                }
            },
            'quality_scoring': {
                'enabled': os.getenv('QUALITY_SCORING_ENABLED', 'true').lower() == 'true',
                'weights': {
                    'kalman': float(os.getenv('QS_WEIGHT_KALMAN', '0.4')),
                    'temporal': float(os.getenv('QS_WEIGHT_TEMPORAL', '0.3')),
                    'source': float(os.getenv('QS_WEIGHT_SOURCE', '0.3'))
                },
                'thresholds': {
                    'outlier_override': float(os.getenv('QS_OUTLIER_OVERRIDE', '0.8')),
                    'acceptance': float(os.getenv('QS_ACCEPTANCE', '0.3'))
                }
            },
            'outlier_detection': {
                'enabled': os.getenv('OUTLIER_DETECTION_ENABLED', 'true').lower() == 'true',
                'iqr_multiplier': float(os.getenv('OUTLIER_IQR_MULTIPLIER', '1.5')),
                'mad_threshold': float(os.getenv('OUTLIER_MAD_THRESHOLD', '3.0'))
            },
            'replay': {
                'enabled': os.getenv('REPLAY_ENABLED', 'false').lower() == 'true',
                'buffer_hours': int(os.getenv('REPLAY_BUFFER_HOURS', '72'))
            },
            'database': {
                'backend': os.getenv('DB_BACKEND', 'memory'),
                'table_name': os.getenv('DB_TABLE_NAME', 'weight-processor-state'),
                'region': os.getenv('AWS_REGION', 'us-east-1')
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'verbose': os.getenv('LOG_VERBOSE', 'false').lower() == 'true'
            }
        }

    @classmethod
    def _load_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        path = Path(config_path)

        if not path.exists():
            # Return minimal defaults if file doesn't exist
            return cls._get_defaults()

        with open(path, 'rb') as f:
            return tomllib.load(f)

    @classmethod
    def _get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {'max_users': 0, 'min_readings': 0},
            'kalman': {
                'enabled': True,
                'adaptive': True,
                'process_noise': 1.0,
                'observation_noise': 4.0
            },
            'quality_scoring': {'enabled': True},
            'database': {'backend': 'memory'},
            'logging': {'level': 'INFO'}
        }

    @classmethod
    def reset_cache(cls):
        """Reset cached configuration (for testing)."""
        cls._cached_config = None