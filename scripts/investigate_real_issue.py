import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from datetime import datetime, timedelta
import numpy as np

# Looking at the screenshot, the issue seems to be with measurements 
# that have small variations being rejected
# Let's test a realistic scenario

scorer = QualityScorer()

# Simulate recent history (all normal weights)
recent_weights = [92.5, 92.8, 93.0, 92.7, 92.9, 93.1, 92.6, 92.8]

print("Testing normal weight variations:")
print("Recent history:", recent_weights)
print(f"Mean: {np.mean(recent_weights):.2f}, Std: {np.std(recent_weights):.2f}")
print("=" * 60)

# Test various measurements that should be accepted
test_cases = [
    (92.4, 92.8, 2.0, "2 hours later, small decrease"),
    (93.2, 92.8, 6.0, "6 hours later, small increase"),  
    (92.0, 92.8, 24.0, "1 day later, moderate decrease"),
    (93.5, 92.8, 48.0, "2 days later, moderate increase"),
    (91.5, 92.8, 72.0, "3 days later, larger decrease"),
]

for weight, prev_weight, hours, description in test_cases:
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=prev_weight,
        time_diff_hours=hours,
        recent_weights=recent_weights,
        user_height_m=1.67
    )
    
    print(f"\n{description}")
    print(f"  Weight: {weight} kg (prev: {prev_weight} kg)")
    print(f"  Overall: {score.overall:.3f} {'✓' if score.accepted else '✗'}")
    print(f"  Components: ", end="")
    for k, v in score.components.items():
        print(f"{k}={v:.2f} ", end="")
    if not score.accepted:
        print(f"\n  REJECTED: {score.rejection_reason}")

print("\n" + "=" * 60)
print("Testing with tighter recent history (less variation):")
recent_tight = [92.8, 92.9, 92.8, 92.7, 92.8, 92.9, 92.8, 92.8]
print("Recent history:", recent_tight)
print(f"Mean: {np.mean(recent_tight):.2f}, Std: {np.std(recent_tight):.2f}")
print("-" * 60)

for weight in [92.5, 92.0, 91.5, 93.5, 94.0]:
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=92.8,
        time_diff_hours=24,
        recent_weights=recent_tight,
        user_height_m=1.67
    )
    
    z_score = abs(weight - np.mean(recent_tight)) / max(np.std(recent_tight), 0.5)
    
    print(f"\nWeight: {weight} kg")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Plausibility: {score.components['plausibility']:.3f}")
    print(f"  Overall: {score.overall:.3f} {'✓' if score.accepted else '✗'}")
    if not score.accepted:
        print(f"  REJECTED: {score.rejection_reason}")
