#!/usr/bin/env python3
"""
Investigate why Kalman doesn't continue after large gap for user 01e5b8da-7c51-458e-8219-ac2be94fda94
"""
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config_loader import load_config
from src.processing.weight_pipeline import WeightProcessingPipeline
from src.core.types import WeightMeasurement

def test_gap_issue():
    config = load_config()
    
    # Critical sequence around the gap
    test_data = [
        # Before gap - around 150-156 kg
        ("2022-05-31 00:00:00", 154.22, "https://connectivehealth.io"),
        ("2022-11-14 00:00:00", 156.48924, "internal-questionnaire"),
        ("2023-03-22 00:00:00", 154.22, "https://connectivehealth.io"),
        
        # After gap - drops to ~83 kg (> 30 day gap, should trigger reset)
        ("2024-07-12 00:00:00", 83.46, "https://connectivehealth.io"),
        ("2024-11-05 00:00:00", 70.31, "https://connectivehealth.io"),
        ("2025-01-14 00:00:00", 72.39, "https://connectivehealth.io"),
        ("2025-05-09 00:00:00", 74.84268, "internal-questionnaire"),
        ("2025-05-22 14:49:00", 74.84274105, "patient-device"),
    ]
    
    user_id = "01e5b8da-7c51-458e-8219-ac2be94fda94"
    pipeline = WeightProcessingPipeline(config)
    
    # Initialize with first few readings
    init_measurements = []
    for date_str, weight, source in test_data[:3]:
        init_measurements.append(WeightMeasurement(
            weight=weight,
            timestamp=datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"),
            source_type=source,
            user_id=user_id
        ))
    
    print(f"Initializing pipeline with {len(init_measurements)} measurements")
    baseline_result = pipeline.initialize_user(init_measurements)
    
    if not baseline_result.success:
        print(f"Baseline establishment failed: {baseline_result.error}")
        # Initialize manually with the average
        avg_weight = sum(m.weight for m in init_measurements) / len(init_measurements)
        print(f"Using manual initialization with average: {avg_weight:.2f} kg")
        from src.filters.robust_kalman import RobustKalmanFilter
        pipeline.kalman = RobustKalmanFilter(
            initial_weight=avg_weight,
            initial_variance=10.0,
            process_noise_weight=0.5,
            process_noise_trend=0.01,
            measurement_noise=5.0,
            outlier_threshold=3.0,
            extreme_outlier_threshold=5.0,
            innovation_window_size=20,
            reset_gap_days=30,
            reset_deviation_threshold=0.5,
            physiological_min=40.0,
            physiological_max=300.0
        )
        pipeline.is_initialized = True
        # Process the init measurements through Kalman
        for m in init_measurements:
            pipeline.kalman.process_measurement(m, source_trust=0.8, robust_mode=True)
    else:
        print(f"Baseline established: {baseline_result.baseline_weight:.2f} kg")
    
    print(f"Kalman state after init: weight={pipeline.kalman.state[0]:.2f} kg\n")
    
    # Process the readings after the gap
    print("=" * 80)
    print("Processing readings after gap:")
    print("=" * 80)
    
    for date_str, weight, source in test_data[3:]:
        measurement = WeightMeasurement(
            weight=weight,
            timestamp=datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"),
            source_type=source,
            user_id=user_id
        )
        
        # Calculate time gap
        if pipeline.kalman.last_timestamp:
            gap_days = (measurement.timestamp - pipeline.kalman.last_timestamp).total_seconds() / 86400
        else:
            gap_days = 0
            
        print(f"\nProcessing: {date_str} - {weight:.2f} kg (gap: {gap_days:.1f} days)")
        print(f"  Kalman state BEFORE: weight={pipeline.kalman.state[0]:.2f} kg")
        
        # Get Layer1 context to see what it will use
        context = pipeline._prepare_layer1_context()
        if 'last_known_state' in context:
            print(f"  Layer1 context weight: {context['last_known_state']['weight']:.2f} kg")
        
        result = pipeline.process_measurement(measurement)
        
        print(f"  Kalman state AFTER: weight={pipeline.kalman.state[0]:.2f} kg")
        
        if result.is_valid:
            print(f"  ✓ ACCEPTED - Kalman estimate: {result.filtered_weight:.2f} kg")
            print(f"    Confidence: {result.confidence:.3f}")
        else:
            print(f"  ✗ REJECTED - Reason: {result.rejection_reason}")
            if result.processing_metadata:
                if 'layer1' in result.processing_metadata:
                    layer1_data = result.processing_metadata['layer1']
                    if 'actual_change_kg' in layer1_data:
                        print(f"    Layer1 saw change: {layer1_data['actual_change_kg']:.2f} kg")
                    if 'change_percent' in layer1_data:
                        print(f"    Layer1 change %: {layer1_data['change_percent']:.1f}%")
                    if 'max_allowed_change_kg' in layer1_data:
                        print(f"    Layer1 max allowed: {layer1_data['max_allowed_change_kg']:.2f} kg")
                if 'kalman' in result.processing_metadata:
                    kalman_data = result.processing_metadata['kalman']
                    if 'state_reset' in kalman_data:
                        print(f"    Kalman reset: {kalman_data['state_reset']}")
                    if 'reset_reason' in kalman_data:
                        print(f"    Reset reason: {kalman_data['reset_reason']}")

if __name__ == "__main__":
    test_gap_issue()