import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from datetime import datetime, timedelta
import numpy as np

# Simulate a sequence of normal measurements from the user
# These are the types of measurements being incorrectly rejected
measurements = [
    (92.99, "2024-12-30 09:38:53"),
    (93.35, "2025-01-03 08:35:02"),  
    (93.35, "2025-01-12 08:45:59"),
    (93.53, "2025-01-12 10:58:28"),
    (93.08, "2025-01-13 08:55:31"),
    (92.44, "2025-01-18 10:47:52"),
    (92.17, "2025-01-19 09:16:13"),
    # Skip 42.22 - that should be rejected
    (92.17, "2025-01-21 17:50:47"),
    (92.08, "2025-01-22 06:57:54"),
    (92.99, "2025-01-22 23:59:59"),
]

scorer = QualityScorer()

print("Testing normal measurements that should be accepted:")
print("=" * 80)

# Build up history
accepted_weights = []

for i, (weight, timestamp) in enumerate(measurements):
    current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    # Get previous measurement
    if i > 0:
        prev_weight, prev_timestamp = measurements[i-1]
        prev_time = datetime.strptime(prev_timestamp, "%Y-%m-%d %H:%M:%S")
        time_diff_hours = (current_time - prev_time).total_seconds() / 3600
    else:
        prev_weight = None
        time_diff_hours = None
    
    # Use recent accepted weights for statistics
    recent_weights = accepted_weights[-20:] if len(accepted_weights) >= 3 else None
    
    score = scorer.calculate_quality_score(
        weight=weight,
        source="patient-device",
        previous_weight=prev_weight,
        time_diff_hours=time_diff_hours,
        recent_weights=recent_weights,
        user_height_m=1.67
    )
    
    status = "✓ ACCEPTED" if score.accepted else "✗ REJECTED"
    print(f"\n{timestamp}: {weight:.2f} kg - {status}")
    print(f"  Overall score: {score.overall:.3f} / {score.threshold:.3f}")
    print(f"  Components: ", end="")
    for comp, val in score.components.items():
        print(f"{comp}={val:.2f} ", end="")
    
    if not score.accepted:
        print(f"\n  Rejection: {score.rejection_reason}")
    
    # Add to history if accepted (simulating what processor would do)
    if score.accepted:
        accepted_weights.append(weight)

print("\n" + "=" * 80)
print("Analysis of the problem:")
print("-" * 40)

# Test with different configurations
configs = [
    {"threshold": 0.6, "use_harmonic_mean": True},  # Current default
    {"threshold": 0.5, "use_harmonic_mean": True},  # Lower threshold
    {"threshold": 0.6, "use_harmonic_mean": False}, # Arithmetic mean
    {"threshold": 0.4, "use_harmonic_mean": True},  # Much lower threshold
]

test_weight = 92.5
prev_weight = 93.0
time_diff = 24.0  # 1 day
recent = [92.0, 92.5, 93.0, 93.5, 93.0, 92.8, 92.6]

print(f"\nTesting weight {test_weight} kg (prev: {prev_weight} kg, 1 day gap)")
print(f"Recent weights: {recent}")
print()

for config in configs:
    scorer_test = QualityScorer(config)
    score = scorer_test.calculate_quality_score(
        weight=test_weight,
        source="patient-device", 
        previous_weight=prev_weight,
        time_diff_hours=time_diff,
        recent_weights=recent,
        user_height_m=1.67
    )
    
    mean_type = "harmonic" if config.get("use_harmonic_mean", True) else "arithmetic"
    print(f"Config: threshold={config['threshold']:.1f}, {mean_type} mean")
    print(f"  Score: {score.overall:.3f} - {'ACCEPTED' if score.accepted else 'REJECTED'}")
    print(f"  Components: safety={score.components['safety']:.2f}, "
          f"plausibility={score.components['plausibility']:.2f}, "
          f"consistency={score.components['consistency']:.2f}, "
          f"reliability={score.components['reliability']:.2f}")
