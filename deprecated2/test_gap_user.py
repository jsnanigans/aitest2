#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config_loader import load_config
from src.processing.weight_pipeline import WeightProcessingPipeline
from src.core.types import WeightReading

def test_user_processing():
    config = load_config()
    
    pipeline = WeightProcessingPipeline(config)
    
    test_data = [
        ("2020-10-21 00:00:00", 154.22, "https://connectivehealth.io"),
        ("2021-10-21 00:00:00", 147.42, "https://connectivehealth.io"),
        ("2022-05-31 00:00:00", 154.22, "https://connectivehealth.io"),
        ("2022-11-14 00:00:00", 156.48924, "internal-questionnaire"),
        ("2023-03-22 00:00:00", 154.22, "https://connectivehealth.io"),
        ("2024-07-12 00:00:00", 83.46, "https://connectivehealth.io"),
        ("2024-11-05 00:00:00", 70.31, "https://connectivehealth.io"),
        ("2025-01-14 00:00:00", 72.39, "https://connectivehealth.io"),
        ("2025-05-09 00:00:00", 74.84268, "internal-questionnaire"),
        ("2025-05-22 14:49:00", 74.84274105, "patient-device"),
        ("2025-07-12 16:56:00", 75.74992579, "patient-device"),
        ("2025-07-13 09:25:00", 75.704566553, "patient-device"),
    ]
    
    user_id = "01e5b8da-7c51-458e-8219-ac2be94fda94"
    
    print(f"\nProcessing user: {user_id}")
    print("=" * 80)
    
    for date_str, weight, source in test_data:
        reading = WeightReading(
            user_id=user_id,
            date=datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"),
            weight=weight,
            source=source
        )
        
        print(f"\nReading: {date_str} - {weight:.2f} kg - {source}")
        result = pipeline.process_reading(reading)
        
        if result:
            print(f"  Kalman estimate: {result.kalman_estimate:.2f} kg")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Layer1 status: {result.layer1_status}")
            print(f"  Kalman status: {result.kalman_status}")
            if result.rejection_reason:
                print(f"  Rejection reason: {result.rejection_reason}")
        else:
            print(f"  ERROR: No result returned")
    
    results = pipeline.finalize_user(user_id)
    if results:
        print("\nFinal Results:")
        print(f"  Total readings: {len(results['readings'])}")
        print(f"  Kalman processed: {sum(1 for r in results['readings'] if r.get('kalman_estimate'))}")
        print(f"  Layer1 rejected: {sum(1 for r in results['readings'] if r.get('layer1_status') == 'rejected')}")

if __name__ == "__main__":
    test_user_processing()