import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from datetime import datetime, timedelta
import numpy as np

# User 03de147f-5e59-49b5-864b-da235f1dab54 data
weights = [
    (91.63, "2020-05-11", "https://connectivehealth.io"),
    (92.53, "2020-05-20", "https://connectivehealth.io"),
    (94.35, "2020-06-10", "https://connectivehealth.io"),
    (94.35, "2020-06-11", "https://connectivehealth.io"),
    (94.35, "2020-06-16", "https://connectivehealth.io"),
    (94.8, "2020-06-25", "https://connectivehealth.io"),
    (96.16, "2020-07-14", "https://connectivehealth.io"),
    (97.52, "2020-08-10", "https://connectivehealth.io"),
    (99.34, "2020-09-04", "https://connectivehealth.io"),
    (102.06, "2020-10-05", "https://connectivehealth.io"),
    (104.06, "2020-11-02", "https://connectivehealth.io"),
    (105.51, "2020-11-18", "https://connectivehealth.io"),
    (107.14, "2020-12-02", "https://connectivehealth.io"),
    (106.59, "2020-12-09", "https://connectivehealth.io"),
    (98.88, "2021-02-10", "https://connectivehealth.io"),
    (84.37, "2021-12-30", "https://connectivehealth.io"),
    (85.05, "2022-01-07", "https://connectivehealth.io"),
    (85.73, "2023-03-29", "https://connectivehealth.io"),
    (88.0, "2024-05-21", "https://connectivehealth.io"),
    (88.0, "2024-05-29", "https://connectivehealth.io"),
    (42.22, "2025-01-21", "patient-device"),  # The problematic value
]

scorer = QualityScorer()

# Test the 42.22kg value
print("Testing 42.22kg measurement from patient-device on 2025-01-21")
print("=" * 60)

# Previous weight would be 88.0kg from 2024-05-29
previous_weight = 88.0
previous_date = datetime(2024, 5, 29)
current_date = datetime(2025, 1, 21)
time_diff = (current_date - previous_date).total_seconds() / 3600  # hours

# Recent weights for statistics (last few measurements)
recent_weights = [98.88, 84.37, 85.05, 85.73, 88.0, 88.0]

score = scorer.calculate_quality_score(
    weight=42.22,
    source="patient-device",
    previous_weight=previous_weight,
    time_diff_hours=time_diff,
    recent_weights=recent_weights,
    user_height_m=1.67
)

print(f"\nOverall Score: {score.overall:.3f}")
print(f"Threshold: {score.threshold:.3f}")
print(f"Accepted: {score.accepted}")
print(f"\nComponent Scores:")
for comp, val in score.components.items():
    print(f"  {comp:12s}: {val:.3f}")

if score.rejection_reason:
    print(f"\nRejection Reason: {score.rejection_reason}")

# Now test consistency calculation in detail
print("\n" + "=" * 60)
print("Detailed Consistency Calculation:")
print(f"  Current weight: 42.22 kg")
print(f"  Previous weight: {previous_weight} kg")
print(f"  Time difference: {time_diff:.1f} hours ({time_diff/24:.1f} days)")
print(f"  Weight difference: {abs(42.22 - previous_weight):.2f} kg")
print(f"  Daily rate: {(abs(42.22 - previous_weight) / time_diff) * 24:.2f} kg/day")

consistency_score = scorer.calculate_consistency_score(
    weight=42.22,
    previous_weight=previous_weight,
    time_diff_hours=time_diff
)
print(f"  Consistency score: {consistency_score:.3f}")

# Test what would be acceptable
print("\n" + "=" * 60)
print("Testing acceptable weight ranges:")
for test_weight in [80, 85, 90, 95, 70, 60, 50]:
    test_score = scorer.calculate_consistency_score(
        weight=test_weight,
        previous_weight=previous_weight,
        time_diff_hours=time_diff
    )
    print(f"  {test_weight:3.0f} kg -> consistency score: {test_score:.3f}")
