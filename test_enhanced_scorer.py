#!/usr/bin/env python3
"""Test the enhanced quality scorer with research-based improvements."""

import sys
sys.path.insert(0, 'src')

from quality_scorer_enhanced import EnhancedQualityScorer
from datetime import datetime, timedelta
import numpy as np

print("=" * 70)
print("ENHANCED QUALITY SCORER TEST")
print("Incorporating research findings:")
print("- Daily fluctuations of 2-3% are normal")
print("- Weekly patterns with 0.35% variation")
print("- MAD-based robust statistics")
print("=" * 70)

scorer = EnhancedQualityScorer()

# Test case: User 03de147f-5e59-49b5-864b-da235f1dab54
print("\nTest Case: Real user with normal variations")
print("-" * 60)

# Simulate recent history (normal weights around 92-93 kg)
recent_weights = [92.99, 93.35, 93.35, 93.53, 93.08, 92.44, 92.17]
baseline = np.median(recent_weights)
print(f"Recent weights: {recent_weights}")
print(f"Baseline (median): {baseline:.2f} kg")
print()

# Test various measurements
test_cases = [
    (92.5, 92.8, 2, datetime(2025, 1, 22, 10, 0), "Small change, 2 hours"),
    (93.5, 92.8, 6, datetime(2025, 1, 22, 14, 0), "Moderate change, 6 hours"),
    (94.0, 92.8, 24, datetime(2025, 1, 23, 8, 0), "1.2kg change, 1 day"),
    (42.22, 92.17, 56, datetime(2025, 1, 21, 17, 33), "Outlier: 42.22 kg"),
    (92.17, 42.22, 0.3, datetime(2025, 1, 21, 17, 50), "After outlier"),
    (93.8, 92.8, 72, datetime(2025, 1, 26, 8, 0), "Weekend peak (Saturday)"),
    (94.2, 93.8, 24, datetime(2025, 1, 27, 8, 0), "Sunday peak"),
    (93.5, 94.2, 24, datetime(2025, 1, 28, 8, 0), "Monday (post-weekend)"),
]

print("Test Results:")
print("-" * 60)
for weight, prev, hours, timestamp, description in test_cases:
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=prev,
        time_diff_hours=hours,
        recent_weights=recent_weights,
        user_height_m=1.67,
        timestamp=timestamp
    )
    
    status = "✓ ACCEPT" if score.accepted else "✗ REJECT"
    weight_change = weight - prev
    change_percent = (abs(weight_change) / baseline) * 100
    
    print(f"\n{description}:")
    print(f"  {prev:.1f} → {weight:.1f} kg ({weight_change:+.1f} kg, {change_percent:.1f}%)")
    print(f"  Time gap: {hours}h, Day: {timestamp.strftime('%A')}")
    print(f"  Score: {score.overall:.3f} {status}")
    print(f"  Components: ", end="")
    for k, v in score.components.items():
        print(f"{k[0].upper()}={v:.2f} ", end="")
    
    if score.confidence_interval:
        lower, upper = score.confidence_interval
        in_range = "✓" if lower <= weight <= upper else "✗"
        print(f"\n  Expected range: {lower:.1f}-{upper:.1f} kg {in_range}")

print("\n" + "=" * 70)
print("PERCENTAGE-BASED THRESHOLDS TEST")
print("-" * 60)

# Test with different body weights
for baseline_weight in [60, 90, 120]:
    print(f"\nBaseline weight: {baseline_weight} kg")
    print("2kg change at different time intervals:")
    
    for hours in [1, 6, 24, 168]:
        score = scorer.calculate_consistency_score_enhanced(
            weight=baseline_weight + 2,
            previous_weight=baseline_weight,
            time_diff_hours=hours,
            baseline_weight=baseline_weight
        )
        
        change_percent = (2 / baseline_weight) * 100
        time_label = f"{hours}h" if hours < 24 else f"{hours/24:.0f}d"
        status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
        
        print(f"  {time_label:4s}: {change_percent:.1f}% change → score {score:.3f} {status}")

print("\n" + "=" * 70)
print("KEY IMPROVEMENTS:")
print("-" * 60)
print("1. Percentage-based thresholds scale with body weight")
print("2. MAD (Median Absolute Deviation) for robust statistics")
print("3. Weekly pattern recognition (weekends allow more variation)")
print("4. Research-validated normal ranges (2-3% daily)")
print("5. Confidence intervals for expected weight range")
