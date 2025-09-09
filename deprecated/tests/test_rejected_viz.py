#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime, timedelta
from src.visualization.dashboard import create_unified_visualization

# Create test data with rejected readings
test_user_id = "test-user-rejected"
test_data = {
    "total_readings": 50,
    "outliers": 5,
    "average_confidence": 0.85,
    "weight_mean": 75.0,
    "weight_std": 2.5,
    "weight_range": 12.0,
    "days_spanned": 100,
    "baseline_established": True,
    "baseline_weight": 74.5,
    "baseline_confidence": "high",
    "baseline_readings_count": 5,
    "baseline_outliers_removed": 1,
    "baseline_std": 0.8,
    "baseline_mad": 0.6,
    "kalman_filter_initialized": True,
    "kalman_summary": {
        "final_filtered_weight": 75.2,
        "mean_trend_kg_per_day": 0.01,
        "mean_uncertainty": 0.5,
        "validation": {
            "rejected_count": 8,
            "accepted_count": 42
        }
    },
    "time_series": [],
    "all_readings": [],
    "kalman_time_series": []
}

# Generate sample data from 2024-11 to 2025-03
base_date = datetime(2024, 11, 1)
base_weight = 74.5

# Generate readings with some rejections
for i in range(50):
    date = base_date + timedelta(days=i*2)
    weight = base_weight + (i * 0.01) + (0.5 if i % 7 == 0 else 0)
    
    # Mark some as rejected (every 5th reading in 2025)
    is_rejected = (date >= datetime(2025, 1, 1) and i % 5 == 0)
    
    reading = {
        "date": date.isoformat(),
        "weight": weight,
        "source": "patient-device" if i % 3 == 0 else "care-team-upload",
        "confidence": 0.95 if not is_rejected else 0.4
    }
    
    # Add to all_readings
    test_data["all_readings"].append({
        "date": date.isoformat(),
        "weight": weight if not is_rejected else weight + 10,  # Rejected readings are outliers
        "source": reading["source"],
        "is_rejected": is_rejected,
        "rejection_reason": "Outlier detected" if is_rejected else None
    })
    
    # Add to time_series only if not rejected
    if not is_rejected:
        test_data["time_series"].append(reading)
    
    # Add Kalman data
    if i >= 5:  # Start Kalman after baseline
        kalman_entry = {
            "date": date.isoformat(),
            "kalman_filtered": weight - 0.2,  # Slightly smoothed
            "kalman_uncertainty": 0.5,
            "innovation": weight - (weight - 0.2),
            "measurement_accepted": not is_rejected
        }
        test_data["kalman_time_series"].append(kalman_entry)
        
        # Add to time_series entry
        if not is_rejected and test_data["time_series"]:
            test_data["time_series"][-1]["kalman_filtered"] = weight - 0.2
            test_data["time_series"][-1]["kalman_uncertainty"] = 0.5

# Create visualization
output_dir = Path("output/test_rejected_viz")
output_dir.mkdir(exist_ok=True, parents=True)

config = {
    "enable_visualization": True,
    "use_all_readings_for_viz": True  # This enables showing rejected points
}

print(f"Creating visualization for test user with {len(test_data['all_readings'])} total readings")
print(f"  - Accepted: {len(test_data['time_series'])}")
print(f"  - Rejected: {len(test_data['all_readings']) - len(test_data['time_series'])}")

create_unified_visualization(test_user_id, test_data, output_dir, config)

print(f"\nVisualization created: {output_dir}/{test_user_id}.png")
print("\nCheck the '2025 Validated Data (Zoomed)' chart - it should now show rejected points in red!")