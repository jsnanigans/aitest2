#!/usr/bin/env python3
"""Debug script to trace the replay scenario step by step."""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "weight_values"))

from weight_values.src.aws.services.weight_processor_service import WeightProcessorService
from weight_values.src.aws.api.models import Measurement
from weight_values.src.core.database.database import ProcessorStateDB
from weight_values.src.aws.config.config_manager import ConfigManager

def main():
    # Test data
    user_id = "TEST-USER-001"
    measurements_data = [
        ("test-001", 99.0, "2025-01-01T10:00:00Z"),
        ("test-002", 106.0, "2025-01-20T15:00:00Z"),
        ("test-003", 100.0, "2025-01-20T15:20:00Z"),
        ("test-004", 98.0, "2025-01-25T12:00:00Z"),
    ]

    measurements = []
    for mid, weight, timestamp_str in measurements_data:
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        measurement = Measurement(
            uuid=mid,
            weight=weight,
            unit="kg",
            effectiveDateTime=timestamp,
            source="https://api.iglucose.com",
            metadata={}
        )
        measurements.append(measurement)

    # Initialize service
    state_store = ProcessorStateDB()
    config = ConfigManager.load_config(source="file")
    config["database"]["backend"] = "memory"
    service = WeightProcessorService(state_store=state_store, config=config)

    print("=" * 80)
    print("PYTHON REPLAY SCENARIO DEBUG")
    print("=" * 80)
    print("\nExpectations:")
    print("  - 106 kg should be REJECTED (outlier: +7 kg jump from 99 kg)")
    print("  - 100 kg should be ACCEPTED (reasonable: only +1 kg from 99 kg)")
    print("\n" + "=" * 80)

    # Process measurements
    response = service.process_batch(user_id, measurements)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i, result in enumerate(response.results):
        measurement = measurements[i]
        status = "✓ ACCEPTED" if result.accepted else "✗ REJECTED"
        mid = measurements_data[i][0]  # Get ID from original data
        weight = measurement.weight_value
        print(f"\n[{mid}] {weight} kg - {status}")
        print(f"  Timestamp: {measurement.measured_at.isoformat()}")
        quality_str = f"{result.quality_score:.4f}" if result.quality_score is not None else "N/A"
        kalman_str = f"{result.kalman_estimate:.2f}" if result.kalman_estimate is not None else "N/A"
        print(f"  Quality Score: {quality_str}")
        print(f"  Kalman Estimate: {kalman_str} kg")
        if not result.accepted and result.rejection_reason:
            print(f"  Rejection Reason: {result.rejection_reason}")
        if result.quality_components:
            print(f"  Quality Components:")
            for component, score in result.quality_components.items():
                print(f"    - {component}: {score:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {response.measurements_processed}")
    print(f"Accepted: {response.measurements_accepted}")
    print(f"Rejected: {response.measurements_rejected}")

    # Check if results match expectations
    test_002_result = response.results[1]  # 106 kg
    test_003_result = response.results[2]  # 100 kg

    print("\n" + "=" * 80)
    print("EXPECTATION CHECK")
    print("=" * 80)

    if not test_002_result.accepted:
        print("✓ PASS: 106 kg was rejected (as expected)")
    else:
        print("✗ FAIL: 106 kg was accepted (should be rejected)")

    if test_003_result.accepted:
        print("✓ PASS: 100 kg was accepted (as expected)")
    else:
        print("✗ FAIL: 100 kg was rejected (should be accepted)")

if __name__ == "__main__":
    main()
