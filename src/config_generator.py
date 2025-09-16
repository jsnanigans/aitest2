"""
Configuration Generator - Converts user-friendly profiles to full config.toml
"""
import toml
from pathlib import Path
from typing import Dict, Any, Optional
import sys


class ConfigGenerator:
    """Generates full configuration from user-friendly profiles."""

    def __init__(self, profiles_path: str = "config_profiles.toml"):
        """Initialize with profiles configuration."""
        self.profiles_path = Path(profiles_path)
        self.profiles_config = toml.load(self.profiles_path)

    def generate_config(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate full configuration from active profile."""
        active_profile = self.profiles_config.get("active_profile", "balanced")

        if active_profile not in self.profiles_config.get("profiles", {}):
            raise ValueError(f"Unknown profile: {active_profile}")

        profile = self.profiles_config["profiles"][active_profile]
        user_settings = self.profiles_config.get("user_settings", {})
        advanced = self.profiles_config.get("advanced", {})
        mappings = self.profiles_config.get("mappings", {})

        # Build full configuration
        config = self._build_base_config(user_settings)

        # Apply profile mappings
        self._apply_filtering_strength(config, profile, mappings)
        self._apply_adaptation_speed(config, profile, mappings)
        self._apply_trust_settings(config, profile, mappings)
        self._apply_gap_sensitivity(config, profile, mappings)
        self._apply_quality_settings(config, profile)

        # Apply advanced overrides
        self._apply_advanced_overrides(config, advanced)

        # Apply visualization settings
        self._apply_visualization_settings(config, user_settings)

        if output_path:
            with open(output_path, 'w') as f:
                toml.dump(config, f)

        return config

    def _build_base_config(self, user_settings: Dict) -> Dict[str, Any]:
        """Build base configuration structure."""
        return {
            "data": {
                "csv_file": user_settings.get("input_file", "./data/weights.csv"),
                "output_dir": user_settings.get("output_directory", "output"),
                "max_users": user_settings.get("max_users_to_process", 200),
                "user_offset": 0,
                "min_readings": user_settings.get("minimum_readings_required", 20),
                "min_date": user_settings.get("start_date", "2015-01-01"),
                "max_date": user_settings.get("end_date", "2026-01-01"),
                "export_database": True
            },
            "processing": {},
            "kalman": {
                "initial_variance": 0.361,
                "transition_covariance_weight": 0.016,
                "transition_covariance_trend": 0.0001,
                "observation_covariance": 3.4,
                "reset": {
                    "initial": {},
                    "hard": {},
                    "soft": {}
                }
            },
            "visualization": {},
            "adaptive_noise": {
                "enabled": True,
                "default_multiplier": 1.5
            },
            "retrospective": {
                "enabled": True,
                "buffer_hours": 1,
                "trigger_mode": "time_based",
                "max_buffer_measurements": 100,
                "state_history_limit": 100,
                "outlier_detection": {
                    "iqr_multiplier": 1.5,
                    "z_score_threshold": 3.0,
                    "temporal_max_change_percent": 0.50,
                    "min_measurements_for_analysis": 2
                },
                "safety": {
                    "max_processing_time_seconds": 60,
                    "require_rollback_confirmation": False,
                    "preserve_immediate_results": True
                }
            },
            "logging": {
                "progress_interval": 10000,
                "timestamp_format": "test_no_date"
            },
            "quality_scoring": {
                "enabled": True,
                "use_harmonic_mean": True,
                "component_weights": {}
            }
        }

    def _apply_filtering_strength(self, config: Dict, profile: Dict, mappings: Dict):
        """Apply filtering strength settings."""
        strength = profile.get("filtering_strength", "moderate")
        strength_map = mappings.get("filtering_strength", {}).get(strength, {})

        config["processing"]["extreme_threshold"] = strength_map.get("extreme_threshold", 0.15)
        config["quality_scoring"]["threshold"] = strength_map.get("quality_threshold", 0.6)

    def _apply_adaptation_speed(self, config: Dict, profile: Dict, mappings: Dict):
        """Apply adaptation speed settings."""
        speed = profile.get("adaptation_speed", "moderate")
        speed_map = mappings.get("adaptation_speed", {}).get(speed, {})

        # Initial reset
        config["kalman"]["reset"]["initial"] = {
            "enabled": True,
            "initial_variance_multiplier": 10,
            "weight_noise_multiplier": 50,
            "trend_noise_multiplier": 50,
            "observation_noise_multiplier": 20,
            "adaptation_measurements": speed_map.get("initial_adapt", 10),
            "adaptation_days": speed_map.get("initial_adapt", 10),
            "adaptation_decay_rate": speed_map.get("decay_rate", 2.5),
            "quality_acceptance_threshold": 0.45,
            "quality_safety_weight": 0.45,
            "quality_plausibility_weight": 0.10,
            "quality_consistency_weight": 0.10,
            "quality_reliability_weight": 0.35
        }

        # Hard reset
        config["kalman"]["reset"]["hard"] = {
            "enabled": True,
            "gap_threshold_days": 30,
            "initial_variance_multiplier": 3,
            "weight_noise_multiplier": 5,
            "trend_noise_multiplier": 50,
            "observation_noise_multiplier": 0.7,
            "adaptation_measurements": speed_map.get("hard_adapt", 10),
            "adaptation_days": 7,
            "adaptation_decay_rate": speed_map.get("decay_rate", 2.5),
            "quality_acceptance_threshold": 0.45,
            "quality_safety_weight": 0.45,
            "quality_plausibility_weight": 0.10,
            "quality_consistency_weight": 0.10,
            "quality_reliability_weight": 0.35
        }

        # Soft reset
        config["kalman"]["reset"]["soft"] = {
            "enabled": True,
            "min_weight_change_kg": 5,
            "cooldown_days": 3,
            "trigger_sources": ["questionnaire", "patient-upload", "care-team-upload"],
            "initial_variance_multiplier": 2,
            "weight_noise_multiplier": 5,
            "trend_noise_multiplier": 20,
            "observation_noise_multiplier": 0.7,
            "adaptation_measurements": speed_map.get("soft_adapt", 15),
            "adaptation_days": 10,
            "adaptation_decay_rate": speed_map.get("decay_rate", 2.5),
            "quality_acceptance_threshold": 0.4,
            "quality_safety_weight": 0.40,
            "quality_plausibility_weight": 0.15,
            "quality_consistency_weight": 0.15,
            "quality_reliability_weight": 0.30
        }

    def _apply_trust_settings(self, config: Dict, profile: Dict, mappings: Dict):
        """Apply manual entry trust settings."""
        trust = profile.get("trust_manual_entries", "moderate")
        trust_map = mappings.get("trust_manual_entries", {}).get(trust, {})

        if "soft_reset_enabled" in trust_map:
            config["kalman"]["reset"]["soft"]["enabled"] = trust_map["soft_reset_enabled"]

        if "min_change" in trust_map:
            config["kalman"]["reset"]["soft"]["min_weight_change_kg"] = trust_map["min_change"]

        if "sources" in trust_map:
            config["kalman"]["reset"]["soft"]["trigger_sources"] = trust_map["sources"]

    def _apply_gap_sensitivity(self, config: Dict, profile: Dict, mappings: Dict):
        """Apply gap sensitivity settings."""
        sensitivity = profile.get("gap_sensitivity", "moderate")
        gap_map = mappings.get("gap_sensitivity", {}).get(sensitivity, {})

        if "gap_days" in gap_map:
            config["kalman"]["reset"]["hard"]["gap_threshold_days"] = gap_map["gap_days"]

    def _apply_quality_settings(self, config: Dict, profile: Dict):
        """Apply quality scoring settings."""
        threshold = profile.get("quality_threshold", 0.6)
        config["quality_scoring"]["threshold"] = threshold

        # Set component weights based on profile
        if profile.get("filtering_strength") == "strict":
            weights = {
                "safety": 0.40,
                "plausibility": 0.25,
                "consistency": 0.25,
                "reliability": 0.10
            }
        elif profile.get("filtering_strength") == "lenient":
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

    def _apply_advanced_overrides(self, config: Dict, advanced: Dict):
        """Apply advanced user overrides."""
        if "extreme_deviation_percent" in advanced:
            config["processing"]["extreme_threshold"] = advanced["extreme_deviation_percent"] / 100

        if "gap_threshold_days" in advanced:
            config["kalman"]["reset"]["hard"]["gap_threshold_days"] = advanced["gap_threshold_days"]

        if "manual_entry_change_threshold_kg" in advanced:
            config["kalman"]["reset"]["soft"]["min_weight_change_kg"] = advanced["manual_entry_change_threshold_kg"]

    def _apply_visualization_settings(self, config: Dict, user_settings: Dict):
        """Apply visualization settings."""
        enabled = user_settings.get("generate_charts", True)
        detail_level = user_settings.get("chart_detail_level", "normal")

        config["visualization"] = {
            "enabled": enabled,
            "use_enhanced": detail_level in ["normal", "detailed"],
            "verbosity": detail_level,
            "markers": {
                "show_source_icons": detail_level in ["normal", "detailed"],
                "show_source_legend": detail_level in ["normal", "detailed"],
                "show_reset_markers": detail_level == "detailed",
                "reset_marker_color": "#FF6600",
                "reset_marker_opacity": 0.2,
                "reset_marker_width": 1,
                "reset_marker_style": "dot"
            },
            "rejection": {
                "show_severity_colors": detail_level in ["normal", "detailed"],
                "group_by_severity": detail_level == "detailed"
            },
            "reset": {
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
        }


def main():
    """Generate config from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate full config from user-friendly profiles")
    parser.add_argument("--profiles", default="config_profiles.toml", help="Path to profiles config")
    parser.add_argument("--output", default="config.toml", help="Output path for generated config")
    parser.add_argument("--show", action="store_true", help="Print generated config to stdout")

    args = parser.parse_args()

    generator = ConfigGenerator(args.profiles)
    config = generator.generate_config(args.output if not args.show else None)

    if args.show:
        import pprint
        pprint.pprint(config)
    else:
        print(f"Generated config saved to: {args.output}")
        print(f"Active profile: {generator.profiles_config.get('active_profile', 'balanced')}")


if __name__ == "__main__":
    main()