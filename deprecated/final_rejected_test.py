#!/usr/bin/env python3
"""
Final comprehensive test of rejected values display in 2025 zoomed chart
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from src.visualization.dashboard import create_unified_visualization

# Create realistic test data
test_user_id = "final-test-rejected"
test_data = {
    "total_readings": 40,
    "outliers": 8,
    "average_confidence": 0.82,
    "weight_mean": 85.0,
    "weight_std": 2.8,
    "weight_range": 25.0,  # Large range due to outliers
    "days_spanned": 90,
    "baseline_established": True,
    "baseline_weight": 84.0,
    "baseline_confidence": "high",
    "baseline_readings_count": 7,
    "kalman_filter_initialized": True,
    "kalman_summary": {
        "final_filtered_weight": 85.5,
        "mean_trend_kg_per_day": 0.015,
        "mean_uncertainty": 0.45,
        "validation": {
            "rejected_count": 8,
            "accepted_count": 32
        }
    },
    "time_series": [],
    "all_readings": [],
    "kalman_time_series": []
}

# Generate mixed data from Dec 2024 to March 2025
base_date = datetime(2024, 12, 1)
base_weight = 84.0

# Rejection patterns:
# - Some outliers in Dec 2024 (historical)
# - More outliers in Jan-Feb 2025 (to show in zoomed view)
# - Mix of sources

for i in range(40):
    date = base_date + timedelta(days=i*2.25)  # ~2.25 days between readings
    normal_weight = base_weight + (i * 0.015) + ((i % 10) * 0.1)  # Some natural variation
    
    # Determine if rejected based on patterns
    if date.month == 12 and i % 8 == 0:  # Some Dec outliers
        is_rejected = True
        weight = normal_weight + 20.0
        rejection_reason = "Historical outlier"
    elif date.year == 2025 and date.month == 1 and i % 6 == 0:  # Jan outliers
        is_rejected = True
        weight = normal_weight - 15.0  # Some are low outliers
        rejection_reason = "Low outlier (possible error)"
    elif date.year == 2025 and date.month == 2 and i % 7 == 0:  # Feb outliers
        is_rejected = True
        weight = normal_weight + 18.0
        rejection_reason = "High outlier (possible scale error)"
    else:
        is_rejected = False
        weight = normal_weight
        rejection_reason = None
    
    # Vary sources
    sources = ["patient-device", "care-team-upload", "patient-upload", "internal-questionnaire"]
    source = sources[i % len(sources)]
    
    # Create all_readings entry
    all_reading = {
        "date": date.isoformat(),
        "weight": weight,
        "source": source,
        "is_rejected": is_rejected,
        "rejection_reason": rejection_reason
    }
    test_data["all_readings"].append(all_reading)
    
    # Add to time_series only if accepted
    if not is_rejected:
        test_data["time_series"].append({
            "date": date.isoformat(),
            "weight": weight,
            "source": source,
            "confidence": 0.90 + (i % 10) * 0.01,  # Vary confidence
            "kalman_filtered": normal_weight - 0.15,
            "kalman_uncertainty": 0.4 + (i % 5) * 0.02
        })
    
    # Add Kalman data for all readings
    kalman_entry = {
        "date": date.isoformat(),
        "kalman_filtered": normal_weight - 0.15,
        "kalman_uncertainty": 0.4 + (i % 5) * 0.02,
        "innovation": weight - (normal_weight - 0.15),
        "measurement_accepted": not is_rejected
    }
    test_data["kalman_time_series"].append(kalman_entry)

# Count rejections in 2025 for verification
rejected_2025 = sum(1 for r in test_data["all_readings"] 
                   if r["is_rejected"] and datetime.fromisoformat(r["date"]).year == 2025)
accepted_2025 = sum(1 for r in test_data["all_readings"] 
                   if not r["is_rejected"] and datetime.fromisoformat(r["date"]).year == 2025)

# Create visualization
output_dir = Path("output/final_rejected_test")
output_dir.mkdir(exist_ok=True, parents=True)

config = {
    "enable_visualization": True,
    "use_all_readings_for_viz": True  # Critical: enables rejected points display
}

print("="*70)
print("FINAL TEST: Rejected Values in 2025 Zoomed Chart")
print("="*70)
print(f"\nData Summary:")
print(f"  Total readings: {len(test_data['all_readings'])}")
print(f"  Date range: Dec 2024 - Mar 2025")
print(f"  Total accepted: {len(test_data['time_series'])}")
print(f"  Total rejected: {len(test_data['all_readings']) - len(test_data['time_series'])}")
print(f"\n2025 Data (shown in zoomed chart):")
print(f"  ✓ Accepted in 2025: {accepted_2025}")
print(f"  ✗ Rejected in 2025: {rejected_2025}")
print("-"*70)

create_unified_visualization(test_user_id, test_data, output_dir, config)

print(f"\n✅ SUCCESS: Visualization created at:")
print(f"   {output_dir}/{test_user_id}.png")
print("\n" + "="*70)
print("VERIFICATION CHECKLIST:")
print("="*70)
print("\n📊 '2025 Validated Data (Zoomed)' Chart Should Show:")
print("  ✓ Rejected points in RED (filled red with dark red edge)")
print(f"  ✓ Title: '2025 Data (Zoomed) - {accepted_2025} validated, {rejected_2025} rejected'")
print(f"  ✓ Legend includes: 'Rejected (n={rejected_2025})'")
print("  ✓ Y-axis range includes all points (both accepted and rejected)")
print("  ✓ Multiple data sources with different markers")
print("  ✓ Kalman filter line if available")
print("\n📈 'Weight Trajectory' Chart Should Show:")
print("  ✓ All historical data (Dec 2024 - Mar 2025)")
print("  ✓ Rejected points throughout the timeline in RED")
print("\n📉 'Kalman Filter Residuals' Chart Should Show:")
print("  ✓ Rejected points with RED outlines")
print("  ✓ Innovation values for all measurements")
print("="*70)