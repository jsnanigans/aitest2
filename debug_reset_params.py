#!/usr/bin/env python3
"""Debug script to check reset_parameters and adaptive Kalman params"""

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python_lib" / "src"))
sys.path.insert(0, str(project_root / "be_implementation_service" / "src"))

# Import from be_implementation_service
from aws.api.models import Measurement
from aws.config.config_manager import ConfigManager

# Import from python_lib
from weight_processor_lib.core.database import InMemoryStore
from weight_processor_lib.core.processing.processor import process_measurement
from weight_processor_lib.core.processing.kalman import get_adaptive_kalman_params, get_reset_timestamp


def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        elif " " in date_str:
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse: {date_str}")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def main():
    csv_file = "/tmp/debug_user_full.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    print("=== Python Reset Parameters Debug ===\n")

    # Load CSV
    measurements = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement = Measurement(
                uuid=row.get("id"),
                weight=float(row.get("value_quantity", "0")),
                unit=row.get("unit", "kg"),
                effectiveDateTime=parse_timestamp(row.get("effective_date_time") or row.get("timestamp")),
                source=row.get("source_type", "unknown"),
            )
            measurements.append(measurement)

    # Sort chronologically
    measurements = sorted(measurements, key=lambda m: m.measured_at)

    # Initialize storage and config
    state_store = InMemoryStore()
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    config["replay"]["buffered_replay_enabled"] = False  # Disable for clearer debugging

    # Process measurements and log reset params for #46-48
    measurement_count = 0

    for measurement in measurements:
        measurement_count += 1

        # Get state BEFORE processing
        state_before = state_store.get_state(user_id)

        result = process_measurement(
            user_id=user_id,
            weight=measurement.weight_value,
            timestamp=measurement.measured_at,
            source=measurement.source,
            config=config,
            unit=measurement.weight_unit,
            db=state_store,
            user_height_m=None
        )

        # Get state AFTER processing
        state_after = state_store.get_state(user_id)

        # Log reset parameters for measurements 46-48
        if 46 <= measurement_count <= 48:
            print(f"\n{'='*80}")
            print(f"MEASUREMENT #{measurement_count}: {measurement.measurement_id}")
            print(f"{'='*80}")
            print(f"Date: {measurement.measured_at.isoformat()}")
            print(f"Weight: {measurement.weight_value} kg")
            print(f"Source: {measurement.source}")

            print(f"\n--- STATE BEFORE ---")
            if state_before and state_before.get("reset_parameters"):
                reset_params = state_before["reset_parameters"]
                print(f"Reset Parameters:")
                print(f"  observation_noise_multiplier: {reset_params.get('observation_noise_multiplier')}")
                print(f"  adaptation_days: {reset_params.get('adaptation_days')}")
                print(f"  adaptation_decay_rate: {reset_params.get('adaptation_decay_rate')}")

                reset_timestamp = get_reset_timestamp(state_before)
                print(f"Reset Timestamp: {reset_timestamp.isoformat() if reset_timestamp else 'N/A'}")

                if reset_timestamp:
                    adaptive_params = get_adaptive_kalman_params(
                        reset_timestamp,
                        measurement.measured_at,
                        config["kalman"],
                        7,
                        state_before
                    )
                    print(f"Adaptive Kalman Params:")
                    print(f"  observation_covariance: {adaptive_params.get('observation_covariance')}")
                    print(f"  transition_covariance_weight: {adaptive_params.get('transition_covariance_weight')}")
            else:
                print(f"Reset Parameters: NOT SET")

            if state_before and "kalman_params" in state_before:
                obs_cov = state_before["kalman_params"]["observation_covariance"]
                print(f"Kalman Params (stored):")
                print(f"  observation_covariance: {obs_cov}")
            else:
                print(f"Kalman Params: NOT SET")

            print(f"\n--- PROCESSING RESULT ---")
            print(f"Accepted: {result.get('accepted')}")
            print(f"Quality Score: {result.get('quality_score', 'N/A')}")

            print(f"\n--- STATE AFTER ---")
            if state_after and state_after.get("reset_parameters"):
                reset_params = state_after["reset_parameters"]
                print(f"Reset Parameters:")
                print(f"  observation_noise_multiplier: {reset_params.get('observation_noise_multiplier')}")
            else:
                print(f"Reset Parameters: NOT SET")

            if state_after and "kalman_params" in state_after:
                obs_cov = state_after["kalman_params"]["observation_covariance"]
                print(f"Kalman Params (stored):")
                print(f"  observation_covariance: {obs_cov}")

    print(f"\n\nProcessed all {measurement_count} measurements")


if __name__ == "__main__":
    main()
