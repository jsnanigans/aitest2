#!/usr/bin/env python3
"""Test the simplified quality scorer with minimal state requirements."""

import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
import numpy as np

print("=" * 70)
print("SIMPLIFIED STATELESS QUALITY SCORER TEST")
print("Minimal state: only previous_weight and time_diff_hours")
print("=" * 70)

scorer = QualityScorer()

# Test cases with only previous weight and time (no recent_weights needed)
test_cases = [
    (92.5, 92.8, 2, "Small change, 2 hours"),
    (93.5, 92.8, 6, "Moderate change, 6 hours"),
    (94.0, 92.8, 24, "1.2kg change, 1 day"),
    (95.5, 92.8, 48, "2.7kg change, 2 days"),
    (42.22, 92.8, 56, "Outlier: 42.22 kg"),
    (88.0, 92.8, 168, "4.8kg change, 1 week"),
]

print("\nTest Results (NO recent_weights required):")
print("-" * 70)
print("Weight  Previous  Hours  Consist  Plausib  Overall  Status")
print("-" * 70)

for weight, prev_weight, hours, description in test_cases:
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=prev_weight,
        time_diff_hours=hours,
        recent_weights=None,  # Not required!
        user_height_m=1.67
    )
    
    status = "✓" if score.accepted else "✗"
    
    print(f"{weight:6.2f}  {prev_weight:8.2f}  {hours:5d}  "
          f"{score.components['consistency']:7.3f}  "
          f"{score.components['plausibility']:7.3f}  "
          f"{score.overall:7.3f}  {status} {description}")

print("\n" + "=" * 70)
print("Testing percentage-based consistency:")
print("-" * 70)

# Test with different baseline weights
for baseline in [60, 90, 120]:
    print(f"\nBaseline: {baseline} kg")
    for change_kg in [1.0, 2.0, 3.0]:
        change_pct = (change_kg / baseline) * 100
        
        # Test 24-hour consistency
        score = scorer.calculate_consistency_score(
            weight=baseline + change_kg,
            previous_weight=baseline,
            time_diff_hours=24
        )
        
        print(f"  {change_kg:.1f}kg change ({change_pct:.1f}%): "
              f"consistency = {score:.3f}")

print("\n" + "=" * 70)
print("KEY ADVANTAGES OF SIMPLIFIED APPROACH:")
print("-" * 70)
print("1. No need to store recent_weights in state")
print("2. Only requires previous_weight and timestamp")
print("3. Still uses percentage-based thresholds")
print("4. Research-based time-aware consistency")
print("5. Maintains stateless architecture")
