import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
import numpy as np

scorer = QualityScorer()

print("Testing IMPROVED consistency scoring:")
print("=" * 70)

# Test cases that were problematic before
test_cases = [
    ("Small change, 1 hour", 92.5, 92.0, 1),
    ("Small change, 2 hours", 92.8, 92.0, 2),
    ("Normal variation, 6 hours", 93.0, 92.0, 6),
    ("Larger change, 1 hour", 94.0, 92.0, 1),
    ("Larger change, 6 hours", 94.0, 92.0, 6),
    ("Big change, 1 hour", 95.0, 92.0, 1),
    ("Big change, 24 hours", 95.0, 92.0, 24),
]

print("Test Case                          Weight  Prev   Hours  Score")
print("-" * 70)
for description, weight, prev, hours in test_cases:
    score = scorer.calculate_consistency_score(weight, prev, hours)
    status = "✓" if score >= 0.6 else "✗"
    print(f"{description:30s}  {weight:6.1f}  {prev:5.1f}  {hours:5d}  {score:.3f} {status}")

print("\n" + "=" * 70)
print("Testing full quality scoring with improved consistency:")
print("-" * 70)

# Simulate a realistic sequence
recent_weights = [92.0, 92.3, 92.1, 92.5, 92.2]

test_measurements = [
    (92.8, 92.5, 2, "2 hours later, small increase"),
    (91.8, 92.8, 1, "1 hour later, 1kg decrease"),
    (93.5, 91.8, 3, "3 hours later, 1.7kg increase"),
]

for weight, prev_weight, hours, desc in test_measurements:
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=prev_weight,
        time_diff_hours=hours,
        recent_weights=recent_weights,
        user_height_m=1.67
    )
    
    print(f"\n{desc}:")
    print(f"  Weight: {weight} kg (prev: {prev_weight} kg)")
    print(f"  Overall: {score.overall:.3f} - {'ACCEPTED' if score.accepted else 'REJECTED'}")
    print(f"  Components: ", end="")
    for k, v in score.components.items():
        print(f"{k}={v:.2f} ", end="")
    if not score.accepted:
        print(f"\n  Reason: {score.rejection_reason}")
