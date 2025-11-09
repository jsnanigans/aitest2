"""Configuration loader for weight processor library."""

import tomllib
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration loading from config.toml."""

    _cached_config: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from config.toml.

        Args:
            config_path: Path to config file (defaults to python_lib/config.toml)

        Returns:
            Configuration dictionary
        """
        # Use cached config if available
        if cls._cached_config is not None:
            return cls._cached_config

        if config_path is None:
            # Default to python_lib/config.toml
            # __file__ is python_lib/src/weight_processor_lib/core/config.py
            # So we go up 4 levels to get to python_lib/
            config_path = Path(__file__).parent.parent.parent.parent / "config.toml"

        config = cls._load_from_file(config_path)

        # Cache the config
        cls._cached_config = config
        return config

    @classmethod
    def _load_from_file(cls, config_path: Path | str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Expected config at python_lib/config.toml"
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
        return config.get("sources", {})

    @classmethod
    def reset_cache(cls):
        """Reset cached configuration (for testing)."""
        cls._cached_config = None
