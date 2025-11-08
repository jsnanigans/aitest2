#!/usr/bin/env python3
"""Detailed Kalman state logging to find divergence point"""

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


def safe_format(val, fmt=".6f"):
    """Safely format a value or return N/A"""
    if val is None:
        return "N/A"
    try:
        return f"{val:{fmt}}"
    except:
        return str(val)


def main():
    csv_file = "/tmp/debug_user_full.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    print("=== Python Detailed State Debug ===\n")

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

    # Process measurements and log state for #47-50
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

        # Log detailed state for measurements 47-50
        if 47 <= measurement_count <= 50:
            print(f"\n{'='*80}")
            print(f"MEASUREMENT #{measurement_count}: {measurement.measurement_id}")
            print(f"{'='*80}")
            print(f"Date: {measurement.measured_at.isoformat()}")
            print(f"Weight: {measurement.weight_value} kg")
            print(f"Source: {measurement.source}")

            if state_before:
                print(f"\n--- STATE BEFORE ---")
                if "kalman_params" in state_before:
                    kp = state_before["kalman_params"]
                    print(f"Kalman State (x):")
                    state_mean = kp.get('state_mean', [None])
                    print(f"  Weight estimate: {safe_format(state_mean[0] if state_mean else None)}")
                    print(f"  Trend (velocity): {safe_format(state_mean[1] if len(state_mean) > 1 else None, '.8f')}")
                    print(f"Kalman Covariance (P):")
                    cov = kp.get('state_covariance', [[None]])
                    print(f"  P[0,0]: {safe_format(cov[0][0] if cov and len(cov) > 0 else None)}")
                    print(f"  P[0,1]: {safe_format(cov[0][1] if cov and len(cov) > 0 and len(cov[0]) > 1 else None)}")
                    print(f"  P[1,0]: {safe_format(cov[1][0] if cov and len(cov) > 1 else None)}")
                    print(f"  P[1,1]: {safe_format(cov[1][1] if cov and len(cov) > 1 and len(cov[1]) > 1 else None)}")
                    print(f"Process Noise (Q):")
                    q = kp.get('process_noise_covariance', [[None]])
                    print(f"  Q[0,0]: {safe_format(q[0][0] if q and len(q) > 0 else None)}")
                    print(f"  Q[1,1]: {safe_format(q[1][1] if q and len(q) > 1 and len(q[1]) > 1 else None, '.8f')}")
                    print(f"Observation Noise (R): {kp.get('observation_covariance', 'N/A')}")
                if "last_raw_weight" in state_before:
                    print(f"Last raw weight: {state_before['last_raw_weight']}")
                if "last_timestamp" in state_before:
                    print(f"Last timestamp: {state_before['last_timestamp']}")

            print(f"\n--- PROCESSING RESULT ---")
            print(f"Accepted: {result.get('accepted')}")
            print(f"Quality Score: {safe_format(result.get('quality_score'))}")
            
            components = result.get('quality_components', {})
            if components:
                print(f"Components:")
                for key in ['kalman_fit', 'temporal_consistency', 'anomaly_detection', 'source_reliability', 'trend_alignment']:
                    print(f"  {key}: {safe_format(components.get(key))}")
            
            print(f"Kalman estimate: {safe_format(result.get('kalman_estimate'))}")
            print(f"Kalman variance: {safe_format(result.get('kalman_variance'))}")
            print(f"Trend (velocity): {safe_format(result.get('trend'), '.8f')}")
            print(f"Innovation: {safe_format(result.get('innovation'))}")
            print(f"Normalized innovation: {safe_format(result.get('normalized_innovation'))}")

            if state_after:
                print(f"\n--- STATE AFTER ---")
                if "kalman_params" in state_after:
                    kp = state_after["kalman_params"]
                    print(f"Kalman State (x):")
                    state_mean = kp.get('state_mean', [None])
                    print(f"  Weight estimate: {safe_format(state_mean[0] if state_mean else None)}")
                    print(f"  Trend (velocity): {safe_format(state_mean[1] if len(state_mean) > 1 else None, '.8f')}")
                    print(f"Kalman Covariance (P):")
                    cov = kp.get('state_covariance', [[None]])
                    print(f"  P[0,0]: {safe_format(cov[0][0] if cov and len(cov) > 0 else None)}")
                    print(f"  P[0,1]: {safe_format(cov[0][1] if cov and len(cov) > 0 and len(cov[0]) > 1 else None)}")
                    print(f"  P[1,0]: {safe_format(cov[1][0] if cov and len(cov) > 1 else None)}")
                    print(f"  P[1,1]: {safe_format(cov[1][1] if cov and len(cov) > 1 and len(cov[1]) > 1 else None)}")

    print(f"\n\nProcessed all {measurement_count} measurements")


if __name__ == "__main__":
    main()
