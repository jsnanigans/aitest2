#!/usr/bin/env python3
"""
Debug script to trace Kalman filter state measurement-by-measurement.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

# Add weight_values to path
sys.path.insert(0, str(Path(__file__).parent / "weight_values"))

from weight_values.src.aws.services.weight_processor_service import WeightProcessorService
from weight_values.src.aws.api.models import Measurement
from weight_values.src.core.database.database import ProcessorStateDB
from weight_values.src.aws.config.config_manager import ConfigManager

def parse_timestamp(date_str: str) -> datetime:
    """Parse timestamp to UTC datetime."""
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def main():
    # Load CSV data
    csv_path = "test_small.csv"
    user_id = "ADC64C0B-CB46-41F9-BDA0-CC11A35942D7"

    measurements = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") != user_id:
                continue

            measurement_id = row.get("id")
            weight = float(row.get("value_quantity"))
            unit = row.get("unit")
            date_str = row.get("effective_date_time")
            source = row.get("source_type", "unknown")
            timestamp = parse_timestamp(date_str)

            measurement = Measurement(
                uuid=measurement_id,
                weight=weight,
                unit=unit,
                effectiveDateTime=timestamp,
                source=source,
                metadata={}
            )
            measurements.append(measurement)

    # Sort by timestamp
    measurements.sort(key=lambda m: m.measured_at)

    # Initialize service
    state_store = ProcessorStateDB()
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    service = WeightProcessorService(state_store=state_store, config=config)

    print("=== PYTHON KALMAN STATE TRACE ===")
    print(f"Processing {len(measurements)} measurements\n")

    # Process one by one with detailed logging
    for i, measurement in enumerate(measurements):
        # Get state BEFORE processing
        state_before = service.state_store.get_state(user_id)

        # Process single measurement
        response = service.process_batch(user_id, [measurement])
        result = response.results[0] if response.results else None

        # Get state AFTER processing
        state_after = service.state_store.get_state(user_id)

        if result:
            kalman_est = result.kalman_estimate if result.kalman_estimate is not None else measurement.weight_value
            innovation = measurement.weight_value - kalman_est if result.kalman_estimate is not None else 0.0
            print(f"[{i:2d}] {measurement.measured_at.isoformat()[:10]} | "
                  f"raw={measurement.weight_value:7.3f} | "
                  f"kalman={kalman_est:7.3f} | "
                  f"innovation={innovation:+7.3f} | "
                  f"accepted={result.accepted}")

            # Show detailed state after processing
            if state_after and state_after.get("last_state") is not None:
                import numpy as np
                last_state = state_after["last_state"]
                last_covariance = state_after["last_covariance"]

                if len(last_state.shape) > 1:
                    state_vec = last_state[-1]
                    cov_mat = last_covariance[-1]
                else:
                    state_vec = last_state
                    cov_mat = last_covariance

                print(f"      state=[{state_vec[0]:7.3f}, {state_vec[1]:+7.5f}] "
                      f"P[0,0]={cov_mat[0,0]:7.4f}")

if __name__ == "__main__":
    main()
