"""Factory for creating application components with dependency injection."""

import os
from typing import Dict, Any, Optional

from ..database.base import StateStore
from ..config.config_manager import ConfigManager
from ..services.weight_processor_service import WeightProcessorService


class ComponentFactory:
    """Factory for creating and managing application components."""

    _instances = {}
    _config = None

    @classmethod
    def get_state_store(cls) -> StateStore:
        """
        Get or create state store instance.
        Always uses DynamoDB for consistency.

        Returns:
            StateStore instance
        """
        cache_key = 'state_store_dynamodb'
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Always use DynamoDB
        try:
            from ..database.dynamodb_store import DynamoDBStateStore
            config = cls.get_config()
            table_name = config.get('database', {}).get('table_name')
            region = config.get('database', {}).get('region')
            instance = DynamoDBStateStore(table_name=table_name, region=region)
        except ImportError as e:
            raise ImportError(
                "DynamoDB (boto3) is required. Install with: pip install boto3"
            ) from e

        # Cache and return
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def get_config(cls, source: str = 'auto', config_path: str = None) -> Dict[str, Any]:
        """
        Get configuration.

        Args:
            source: 'file', 'env', or 'auto'
            config_path: Path to config file (for 'file' source)

        Returns:
            Configuration dictionary
        """
        if cls._config is None:
            cls._config = ConfigManager.load_config(source, config_path)
        return cls._config

    @classmethod
    def get_weight_processor_service(cls, state_store: StateStore = None,
                                    config: Dict[str, Any] = None) -> WeightProcessorService:
        """
        Get or create weight processor service.

        Args:
            state_store: Optional state store instance
            config: Optional configuration

        Returns:
            WeightProcessorService instance
        """
        cache_key = 'weight_processor_service'

        # Return cached instance if available
        if cache_key in cls._instances and state_store is None and config is None:
            return cls._instances[cache_key]

        # Create new instance
        if state_store is None:
            state_store = cls.get_state_store()
        if config is None:
            config = cls.get_config()

        instance = WeightProcessorService(state_store, config)

        # Cache if using defaults
        if state_store is None and config is None:
            cls._instances[cache_key] = instance

        return instance

    @classmethod
    def reset(cls):
        """Reset all cached instances (useful for testing)."""
        cls._instances.clear()
        cls._config = None
        ConfigManager.reset_cache()

    @classmethod
    def create_kalman_filter(cls, config: Dict[str, Any] = None):
        """
        Create a configured Kalman filter instance.

        Args:
            config: Optional configuration override

        Returns:
            Configured Kalman filter
        """
        from ..processing.kalman import AdaptiveKalmanFilter

        if config is None:
            config = cls.get_config()

        kalman_config = config.get('kalman', {})
        return AdaptiveKalmanFilter(
            process_noise=kalman_config.get('process_noise', 1.0),
            observation_noise=kalman_config.get('observation_noise', 4.0),
            adaptive=kalman_config.get('adaptive', True)
        )

    @classmethod
    def create_quality_scorer(cls, config: Dict[str, Any] = None):
        """
        Create a configured quality scorer instance.

        Args:
            config: Optional configuration override

        Returns:
            Configured quality scorer
        """
        from ..processing.unified_quality_scorer import UnifiedQualityScorer

        if config is None:
            config = cls.get_config()

        return UnifiedQualityScorer(config)

    @classmethod
    def create_outlier_detector(cls, config: Dict[str, Any] = None):
        """
        Create a configured outlier detector instance.

        Args:
            config: Optional configuration override

        Returns:
            Configured outlier detector
        """
        from ..processing.outlier_detection import OutlierDetector

        if config is None:
            config = cls.get_config()

        outlier_config = config.get('outlier_detection', {})
        return OutlierDetector(outlier_config)