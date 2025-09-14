import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from datetime import datetime, timedelta
import numpy as np

# Include the outlier in the sequence
measurements = [
    (92.44, "2025-01-18 10:47:52"),
    (92.17, "2025-01-19 09:16:13"),
    (42.22, "2025-01-21 17:33:21"),  # The outlier
    (92.17, "2025-01-21 17:50:47"),  # Just 17 minutes later!
    (92.08, "2025-01-22 06:57:54"),
    (92.99, "2025-01-22 23:59:59"),
    (92.44, "2025-01-23 08:13:57"),
]

scorer = QualityScorer()

print("Testing sequence WITH the 42.22 kg outlier:")
print("=" * 80)

# Simulate what happens if outlier gets into history
scenarios = [
    ("Scenario 1: Outlier NOT in history (correctly rejected)", False),
    ("Scenario 2: Outlier IN history (if it were accepted)", True),
]

for scenario_name, include_outlier in scenarios:
    print(f"\n{scenario_name}")
    print("-" * 60)
    
    # Build history
    if include_outlier:
        recent_weights = [92.44, 92.17, 42.22, 92.17, 92.08]  # Outlier included
    else:
        recent_weights = [92.44, 92.17, 92.17, 92.08, 92.99]  # Normal weights only
    
    print(f"Recent weights in history: {recent_weights}")
    print(f"  Mean: {np.mean(recent_weights):.2f} kg")
    print(f"  Std: {np.std(recent_weights):.2f} kg")
    print()
    
    # Test a normal measurement
    test_weight = 92.5
    score = scorer.calculate_quality_score(
        weight=test_weight,
        source="patient-device",
        previous_weight=92.99,
        time_diff_hours=24,
        recent_weights=recent_weights,
        user_height_m=1.67
    )
    
    print(f"Testing {test_weight} kg:")
    print(f"  Overall score: {score.overall:.3f}")
    print(f"  Plausibility: {score.components['plausibility']:.3f}")
    print(f"  Status: {'ACCEPTED' if score.accepted else 'REJECTED'}")

print("\n" + "=" * 80)
print("The problem is clear:")
print("-" * 40)
print("If the 42.22 kg outlier somehow gets into the recent_weights history,")
print("it corrupts the statistics (mean/std) used for plausibility scoring,")
print("causing subsequent NORMAL measurements to be rejected!")
print()
print("This suggests the issue is in the processor's history management,")
print("not in the quality scorer itself.")
