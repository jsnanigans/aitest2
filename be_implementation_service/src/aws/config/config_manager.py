"""Configuration management for multiple environments."""

import os
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration from multiple sources."""

    _cached_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(
        cls, source: str = "auto", config_path: str = None
    ) -> Dict[str, Any]:
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
        if source == "auto":
            if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
                source = "env"
            else:
                source = "file"

        # Load base configuration from file
        if config_path is None:
            # Default to weight_values/config.toml
            # __file__ is weight_values/src/aws/config/config_manager.py
            # So we go up 4 levels to get to weight_values/
            config_path = Path(__file__).parent.parent.parent.parent / "config.toml"

        config = cls._load_from_file(config_path)

        # Override with environment variables if in Lambda
        if source == "env":
            config = cls._apply_env_overrides(config)

        # Cache the config
        cls._cached_config = config
        return config

    @classmethod
    def _apply_env_overrides(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        # Only override specific values if env vars are set

        # Kalman overrides
        if os.getenv("KALMAN_INITIAL_VARIANCE"):
            config["kalman"]["initial_variance"] = float(os.getenv("KALMAN_INITIAL_VARIANCE"))
        if os.getenv("KALMAN_OBS_COVARIANCE"):
            config["kalman"]["observation_covariance"] = float(os.getenv("KALMAN_OBS_COVARIANCE"))

        # Quality scoring overrides
        if os.getenv("QS_THRESHOLD"):
            config["quality_scoring"]["threshold"] = float(os.getenv("QS_THRESHOLD"))
        if os.getenv("QS_USE_HARMONIC_MEAN"):
            config["quality_scoring"]["use_harmonic_mean"] = os.getenv("QS_USE_HARMONIC_MEAN").lower() == "true"

        # Database overrides
        if os.getenv("DB_BACKEND"):
            config["database"]["backend"] = os.getenv("DB_BACKEND")
        if os.getenv("DB_TABLE_NAME"):
            config["database"]["table_name"] = os.getenv("DB_TABLE_NAME")
        if os.getenv("AWS_REGION"):
            config["database"]["region"] = os.getenv("AWS_REGION")

        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            config["logging"]["level"] = os.getenv("LOG_LEVEL")

        return config

    @classmethod
    def _load_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Expected config at weight_values/config.toml"
            )

        with open(path, "rb") as f:
            return tomllib.load(f)

    @classmethod
    def get_source_profiles(cls, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract source profiles from config.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of source profiles
        """
        sources = config.get("sources", {})
        # Convert TOML section names back to source names
        profiles = {}
        for source_key, profile in sources.items():
            profiles[source_key] = profile
        return profiles

    @classmethod
    def reset_cache(cls):
        """Reset cached configuration (for testing)."""
        cls._cached_config = None
