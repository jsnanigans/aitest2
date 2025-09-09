#!/usr/bin/env python3
"""Test the fix for the 52.1 kg issue with user 03b792f9-b7e2-4e9c-b2aa-723236ce49b0"""

import json
from datetime import datetime
from pathlib import Path

from src.processing.weight_pipeline import WeightProcessingPipeline
from src.core.types import WeightMeasurement
from src.core.config_loader import load_config

def test_specific_user():
    """Test processing for the user with the 52.1 kg anomaly."""
    
    # Load config
    config = load_config('config.toml')
    
    # Initialize pipeline
    pipeline = WeightProcessingPipeline(config)
    
    # Initialize with first measurement as baseline (around 140 kg)
    initial_measurement = WeightMeasurement(
        weight=140.0,
        timestamp=datetime.fromisoformat('2025-06-01T10:00:00'),
        source_type='https://api.iglucose.com'
    )
    pipeline.initialize(initial_measurement, variance=1.0)
    
    # Simulate the specific sequence of measurements
    measurements = [
        # June 6 - normal weight
        WeightMeasurement(
            weight=140.5,
            timestamp=datetime.fromisoformat('2025-06-06T10:47:40'),
            source_type='https://api.iglucose.com'
        ),
        # June 10 - anomalous 52.1 kg
        WeightMeasurement(
            weight=52.1,
            timestamp=datetime.fromisoformat('2025-06-10T16:55:01'),
            source_type='https://api.iglucose.com'
        ),
        # June 18 - back to normal
        WeightMeasurement(
            weight=140.7,
            timestamp=datetime.fromisoformat('2025-06-18T10:59:57'),
            source_type='https://api.iglucose.com'
        ),
    ]
    
    print("Testing weight sequence around 52.1 kg anomaly:")
    print("-" * 60)
    
    results = []
    for i, measurement in enumerate(measurements):
        result = pipeline.process_measurement(measurement)
        results.append(result)
        
        print(f"\nMeasurement {i+1}:")
        print(f"  Date: {measurement.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Input Weight: {measurement.weight:.1f} kg")
        print(f"  Is Valid: {result.is_valid}")
        print(f"  Filtered Weight: {result.filtered_weight:.1f} kg" if result.filtered_weight else f"  Filtered Weight: N/A")
        print(f"  Confidence: {result.confidence:.2f}")
        
        if not result.is_valid:
            print(f"  Rejection Reason: {result.rejection_reason}")
            if hasattr(result, 'layer1_metadata') and result.layer1_metadata:
                meta = result.layer1_metadata
                if 'reason' in meta:
                    print(f"  Details: {meta['reason']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  52.1 kg weight was: {'REJECTED ✓' if not results[1].is_valid else 'ACCEPTED ✗ (BUG NOT FIXED)'}")
    
    return results

if __name__ == "__main__":
    test_specific_user()