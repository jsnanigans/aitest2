"""
Configuration loader that interprets high-level profiles into full config.
"""
from typing import Dict, Any
import tomllib
from pathlib import Path
from src.feature_manager import FeatureManager


class ConfigLoader:
    """Loads and interprets configuration with profile support."""

    # Profile mappings for different settings
    FILTERING_STRENGTH_MAP = {
        "strict": {"extreme_threshold": 0.10, "quality_threshold": 0.70},
        "moderate": {"extreme_threshold": 0.15, "quality_threshold": 0.60},
        "lenient": {"extreme_threshold": 0.25, "quality_threshold": 0.45},
        "source_based": {"extreme_threshold": 0.15, "quality_threshold": 0.55}
    }

    ADAPTATION_SPEED_MAP = {
        "slow": {"initial_adapt": 20, "hard_adapt": 15, "soft_adapt": 20, "decay_rate": 4.0},
        "moderate": {"initial_adapt": 10, "hard_adapt": 10, "soft_adapt": 15, "decay_rate": 2.5},
        "fast": {"initial_adapt": 5, "hard_adapt": 5, "soft_adapt": 10, "decay_rate": 1.0}
    }

    TRUST_MANUAL_MAP = {
        "low": {"soft_reset_enabled": False},
        "moderate": {"soft_reset_enabled": True, "min_change": 5},
        "high": {"soft_reset_enabled": True, "min_change": 3},
        "clinical_only": {"soft_reset_enabled": True, "sources": ["care-team-upload"]}
    }

    GAP_SENSITIVITY_MAP = {
        "low": {"gap_days": 45},
        "moderate": {"gap_days": 30},
        "high": {"gap_days": 21}
    }

    # New reset behavior mappings
    RESET_ON_GAPS_MAP = {
        "aggressive": {"gap_days": 21, "hard_variance_mult": 5, "hard_adapt": 15},
        "moderate": {"gap_days": 30, "hard_variance_mult": 3, "hard_adapt": 10},
        "lenient": {"gap_days": 45, "hard_variance_mult": 2, "hard_adapt": 7}
    }

    RESET_ON_MANUAL_MAP = {
        "disabled": {"enabled": False},
        "sensitive": {"enabled": True, "min_change": 3, "sources": ["questionnaire", "patient-upload", "care-team-upload"]},
        "moderate": {"enabled": True, "min_change": 5, "sources": ["questionnaire", "patient-upload", "care-team-upload"]},
        "clinical": {"enabled": True, "min_change": 5, "sources": ["care-team-upload"]}
    }

    INITIAL_TRUST_MAP = {
        "low": {"initial_adapt": 20, "initial_variance_mult": 15, "initial_decay": 4.0},
        "moderate": {"initial_adapt": 10, "initial_variance_mult": 10, "initial_decay": 2.5},
        "high": {"initial_adapt": 5, "initial_variance_mult": 5, "initial_decay": 1.5},
        "clinical": {"initial_adapt": 10, "initial_variance_mult": 10, "initial_decay": 2.5}
    }

    @classmethod
    def load(cls, config_path: str = "config.toml") -> Dict[str, Any]:
        """Load and interpret configuration file."""
        # Load the raw config
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)

        # Get the active profile
        profile_name = raw_config.get("profile", "balanced")

        # Get the profile definition
        profiles = raw_config.get("profiles", {})
        if profile_name not in profiles:
            print(f"Warning: Unknown profile '{profile_name}', using balanced")
            profile_name = "balanced"

        profile = profiles.get(profile_name, profiles.get("balanced", {}))

        # Start with base config structure
        config = cls._build_base_config(raw_config)

        # Apply profile settings
        cls._apply_profile(config, profile)

        # Apply any explicit overrides from advanced sections
        cls._apply_overrides(config, raw_config)

        # Apply visualization settings
        cls._apply_visualization(config, raw_config)

        # Add feature manager instance
        config["feature_manager"] = FeatureManager(raw_config)

        return config

    @classmethod
    def _build_base_config(cls, raw_config: Dict) -> Dict[str, Any]:
        """Build the base configuration structure."""
        config = {
            "data": raw_config.get("data", {}),
            "processing": {},
            "kalman": {
                "initial_variance": 0.361,
                "transition_covariance_weight": 0.016,
                "transition_covariance_trend": 0.0001,
                "observation_covariance": 3.4,
                "reset": {
                    "initial": {"enabled": True},
                    "hard": {"enabled": True},
                    "soft": {"enabled": True}
                }
            },
            "visualization": raw_config.get("visualization", {"enabled": True}),
            "adaptive_noise": raw_config.get("adaptive_noise", {"enabled": True}),
            "retrospective": raw_config.get("retrospective", {}),
            "logging": raw_config.get("logging", {}),
            "quality_scoring": {
                "enabled": True,
                "use_harmonic_mean": True
            }
        }

        # Copy Kalman base parameters if specified
        if "kalman" in raw_config:
            for key in ["initial_variance", "transition_covariance_weight",
                       "transition_covariance_trend", "observation_covariance"]:
                if key in raw_config["kalman"]:
                    config["kalman"][key] = raw_config["kalman"][key]

        return config

    @classmethod
    def _apply_profile(cls, config: Dict, profile: Dict):
        """Apply profile settings to configuration."""
        # Filtering strength
        strength = profile.get("filtering_strength", "moderate")
        strength_settings = cls.FILTERING_STRENGTH_MAP.get(strength, cls.FILTERING_STRENGTH_MAP["moderate"])
        config["processing"]["extreme_threshold"] = strength_settings["extreme_threshold"]
        config["quality_scoring"]["threshold"] = profile.get("quality_threshold", strength_settings["quality_threshold"])

        # Get reset behavior settings from profile
        reset_gaps = profile.get("reset_on_gaps", "moderate")
        reset_manual = profile.get("reset_on_manual", "moderate")
        initial_trust = profile.get("initial_trust", "moderate")

        # Get mappings
        gaps_settings = cls.RESET_ON_GAPS_MAP.get(reset_gaps, cls.RESET_ON_GAPS_MAP["moderate"])
        manual_settings = cls.RESET_ON_MANUAL_MAP.get(reset_manual, cls.RESET_ON_MANUAL_MAP["moderate"])
        initial_settings = cls.INITIAL_TRUST_MAP.get(initial_trust, cls.INITIAL_TRUST_MAP["moderate"])

        # Apply to initial reset (new user)
        config["kalman"]["reset"]["initial"].update({
            "initial_variance_multiplier": initial_settings["initial_variance_mult"],
            "weight_noise_multiplier": 50,
            "trend_noise_multiplier": 50,
            "observation_noise_multiplier": 20,
            "adaptation_measurements": initial_settings["initial_adapt"],
            "adaptation_days": initial_settings["initial_adapt"],
            "adaptation_decay_rate": initial_settings["initial_decay"],
            "quality_acceptance_threshold": 0.45,
            "quality_safety_weight": 0.45,
            "quality_plausibility_weight": 0.10,
            "quality_consistency_weight": 0.10,
            "quality_reliability_weight": 0.35
        })

        # Apply to hard reset (gaps)
        config["kalman"]["reset"]["hard"].update({
            "gap_threshold_days": gaps_settings["gap_days"],
            "initial_variance_multiplier": gaps_settings["hard_variance_mult"],
            "weight_noise_multiplier": 5,
            "trend_noise_multiplier": 50,
            "observation_noise_multiplier": 0.7,
            "adaptation_measurements": gaps_settings["hard_adapt"],
            "adaptation_days": 7,
            "adaptation_decay_rate": 2.5,
            "quality_acceptance_threshold": 0.45,
            "quality_safety_weight": 0.45,
            "quality_plausibility_weight": 0.10,
            "quality_consistency_weight": 0.10,
            "quality_reliability_weight": 0.35
        })

        # Apply to soft reset (manual entries)
        config["kalman"]["reset"]["soft"]["enabled"] = manual_settings.get("enabled", True)
        if manual_settings.get("enabled", True):
            config["kalman"]["reset"]["soft"].update({
                "min_weight_change_kg": manual_settings.get("min_change", 5),
                "cooldown_days": 3,
                "trigger_sources": manual_settings.get("sources", ["questionnaire", "patient-upload", "care-team-upload"]),
                "initial_variance_multiplier": 2,
                "weight_noise_multiplier": 5,
                "trend_noise_multiplier": 20,
                "observation_noise_multiplier": 0.7,
                "adaptation_measurements": 15,
                "adaptation_days": 10,
                "adaptation_decay_rate": 2.5,
                "quality_acceptance_threshold": 0.4,
                "quality_safety_weight": 0.40,
                "quality_plausibility_weight": 0.15,
                "quality_consistency_weight": 0.15,
                "quality_reliability_weight": 0.30
            })

        # Keep backward compatibility with old settings if they exist
        # (These override the new reset behavior settings)
        if "gap_sensitivity" in profile:
            gap = profile["gap_sensitivity"]
            gap_settings = cls.GAP_SENSITIVITY_MAP.get(gap, cls.GAP_SENSITIVITY_MAP["moderate"])
            config["kalman"]["reset"]["hard"]["gap_threshold_days"] = gap_settings["gap_days"]

        if "trust_manual_entries" in profile:
            trust = profile["trust_manual_entries"]
            trust_settings = cls.TRUST_MANUAL_MAP.get(trust, cls.TRUST_MANUAL_MAP["moderate"])
            if "soft_reset_enabled" in trust_settings:
                config["kalman"]["reset"]["soft"]["enabled"] = trust_settings["soft_reset_enabled"]
            if "min_change" in trust_settings:
                config["kalman"]["reset"]["soft"]["min_weight_change_kg"] = trust_settings["min_change"]
            if "sources" in trust_settings:
                config["kalman"]["reset"]["soft"]["trigger_sources"] = trust_settings["sources"]

        if "adaptation_speed" in profile:
            speed = profile["adaptation_speed"]
            speed_settings = cls.ADAPTATION_SPEED_MAP.get(speed, cls.ADAPTATION_SPEED_MAP["moderate"])
            # Update adaptation measurements based on speed
            config["kalman"]["reset"]["initial"]["adaptation_measurements"] = speed_settings["initial_adapt"]
            config["kalman"]["reset"]["initial"]["adaptation_days"] = speed_settings["initial_adapt"]
            config["kalman"]["reset"]["hard"]["adaptation_measurements"] = speed_settings["hard_adapt"]
            config["kalman"]["reset"]["soft"]["adaptation_measurements"] = speed_settings["soft_adapt"]

        # Quality scoring weights based on filtering strength
        if strength == "strict":
            weights = {
                "safety": 0.40,
                "plausibility": 0.25,
                "consistency": 0.25,
                "reliability": 0.10
            }
        elif strength == "lenient":
            weights = {
                "safety": 0.30,
                "plausibility": 0.20,
                "consistency": 0.20,
                "reliability": 0.30
            }
        else:  # moderate or source_based
            weights = {
                "safety": 0.35,
                "plausibility": 0.25,
                "consistency": 0.25,
                "reliability": 0.15
            }

        config["quality_scoring"]["component_weights"] = weights

    @classmethod
    def _apply_overrides(cls, config: Dict, raw_config: Dict):
        """Apply any explicit overrides from advanced sections."""
        # Processing overrides
        if "processing" in raw_config and "extreme_threshold" in raw_config["processing"]:
            config["processing"]["extreme_threshold"] = raw_config["processing"]["extreme_threshold"]

        # Quality scoring overrides
        if "quality_scoring" in raw_config:
            qs = raw_config["quality_scoring"]
            if "threshold" in qs:
                config["quality_scoring"]["threshold"] = qs["threshold"]
            if "component_weights" in qs:
                config["quality_scoring"]["component_weights"] = qs["component_weights"]

        # Source multipliers
        if "adaptive_noise" in raw_config and "source_multipliers" in raw_config["adaptive_noise"]:
            config["adaptive_noise"]["default_multiplier"] = 1.5
            # The actual multipliers will be read directly by the code

        # Direct Kalman reset overrides (from expert section)
        if "kalman" in raw_config and "reset" in raw_config["kalman"]:
            reset_config = raw_config["kalman"]["reset"]

            # Override initial reset if specified
            if "initial" in reset_config:
                config["kalman"]["reset"]["initial"].update(reset_config["initial"])

            # Override hard reset if specified
            if "hard" in reset_config:
                config["kalman"]["reset"]["hard"].update(reset_config["hard"])

            # Override soft reset if specified
            if "soft" in reset_config:
                config["kalman"]["reset"]["soft"].update(reset_config["soft"])

    @classmethod
    def _apply_visualization(cls, config: Dict, raw_config: Dict):
        """Apply visualization settings based on detail level."""
        viz = raw_config.get("visualization", {})
        detail_level = viz.get("detail_level", "normal")

        config["visualization"]["use_enhanced"] = detail_level in ["normal", "detailed"]
        config["visualization"]["verbosity"] = detail_level

        # Markers settings
        config["visualization"]["markers"] = {
            "show_source_icons": detail_level in ["normal", "detailed"],
            "show_source_legend": detail_level in ["normal", "detailed"],
            "show_reset_markers": detail_level == "detailed",
            "reset_marker_color": "#FF6600",
            "reset_marker_opacity": 0.2,
            "reset_marker_width": 1,
            "reset_marker_style": "dot"
        }

        # Rejection settings
        config["visualization"]["rejection"] = {
            "show_severity_colors": detail_level in ["normal", "detailed"],
            "group_by_severity": detail_level == "detailed"
        }

        # Reset settings
        config["visualization"]["reset"] = {
            "show_reset_lines": detail_level == "detailed",
            "show_gap_regions": detail_level == "detailed",
            "show_gap_labels": detail_level == "detailed",
            "reset_line_color": "#FF6600",
            "reset_line_style": "dash",
            "reset_line_width": 2,
            "gap_region_color": "#F0F0F0",
            "gap_region_opacity": 0.5,
            "gap_region_pattern": "diagonal",
            "show_transition_markers": detail_level == "detailed",
            "transition_marker_size": 10
        }


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration with profile interpretation."""
    return ConfigLoader.load(config_path)