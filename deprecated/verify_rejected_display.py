#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime, timedelta
from src.visualization.dashboard import create_unified_visualization

# Create test data specifically for 2025 with clear rejected readings
test_user_id = "test-2025-rejected"
test_data = {
    "total_readings": 30,
    "outliers": 6,
    "average_confidence": 0.80,
    "weight_mean": 80.0,
    "weight_std": 3.0,
    "weight_range": 15.0,
    "days_spanned": 60,
    "baseline_established": True,
    "baseline_weight": 79.5,
    "baseline_confidence": "high",
    "baseline_readings_count": 5,
    "kalman_filter_initialized": True,
    "kalman_summary": {
        "final_filtered_weight": 80.2,
        "mean_trend_kg_per_day": 0.02,
        "mean_uncertainty": 0.4,
        "validation": {
            "rejected_count": 6,
            "accepted_count": 24
        }
    },
    "time_series": [],
    "all_readings": [],
    "kalman_time_series": []
}

# Generate data only for 2025 to make it clear
base_date = datetime(2025, 1, 1)
base_weight = 80.0

# Generate 30 readings with clear rejection pattern
for i in range(30):
    date = base_date + timedelta(days=i*2)
    normal_weight = base_weight + (i * 0.02)
    
    # Every 5th reading is rejected as an outlier
    is_rejected = (i % 5 == 0 and i > 0)
    
    # Create the reading
    if is_rejected:
        # Rejected readings are outliers (much higher weight)
        weight = normal_weight + 15.0
        confidence = 0.3
        rejection_reason = "Extreme outlier"
    else:
        weight = normal_weight
        confidence = 0.95
        rejection_reason = None
    
    # Add to all_readings (includes both accepted and rejected)
    all_reading = {
        "date": date.isoformat(),
        "weight": weight,
        "source": "patient-device" if i % 3 == 0 else "care-team-upload",
        "is_rejected": is_rejected,
        "rejection_reason": rejection_reason
    }
    test_data["all_readings"].append(all_reading)
    
    # Add to time_series only if not rejected
    if not is_rejected:
        test_data["time_series"].append({
            "date": date.isoformat(),
            "weight": normal_weight,
            "source": all_reading["source"],
            "confidence": confidence,
            "kalman_filtered": normal_weight - 0.1,
            "kalman_uncertainty": 0.4
        })
    
    # Add Kalman data
    kalman_entry = {
        "date": date.isoformat(),
        "kalman_filtered": normal_weight - 0.1,
        "kalman_uncertainty": 0.4,
        "innovation": weight - (normal_weight - 0.1),
        "measurement_accepted": not is_rejected
    }
    test_data["kalman_time_series"].append(kalman_entry)

# Create visualization
output_dir = Path("output/test_2025_rejected")
output_dir.mkdir(exist_ok=True, parents=True)

config = {
    "enable_visualization": True,
    "use_all_readings_for_viz": True  # Enable showing rejected points
}

print("="*60)
print("TEST: 2025 Data with Rejected Points Visualization")
print("="*60)
print(f"\nTotal readings in 2025: {len(test_data['all_readings'])}")
print(f"  ✓ Accepted: {len(test_data['time_series'])} (shown in green/blue)")
print(f"  ✗ Rejected: {len(test_data['all_readings']) - len(test_data['time_series'])} (should show in RED)")
print("\nRejected readings pattern: Every 5th reading is an outlier (+15kg)")
print("-"*60)

create_unified_visualization(test_user_id, test_data, output_dir, config)

print(f"\n✓ Visualization created: {output_dir}/{test_user_id}.png")
print("\n" + "="*60)
print("CHECK THE FOLLOWING IN THE VISUALIZATION:")
print("="*60)
print("1. 'Weight Trajectory' chart:")
print("   - Should show rejected points in RED")
print("   - Accepted points in BLUE")
print("\n2. '2025 Validated Data (Zoomed)' chart (NEW FEATURE):")
print("   - Should NOW show rejected points in RED")
print("   - Accepted points with confidence colors")
print("   - Title should say '24 validated, 6 rejected'")
print("   - Legend should include 'Rejected (n=6)'")
print("\n3. 'Kalman Filter Residuals' chart:")
print("   - Should show rejected points with RED outline")
print("="*60)