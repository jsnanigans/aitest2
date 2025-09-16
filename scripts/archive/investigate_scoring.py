import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from datetime import datetime, timedelta
import numpy as np

scorer = QualityScorer()

# Test plausibility scoring
recent_weights = [98.88, 84.37, 85.05, 85.73, 88.0, 88.0]
print("Recent weights for plausibility check:", recent_weights)
print(f"Mean: {np.mean(recent_weights):.2f} kg")
print(f"Std: {np.std(recent_weights):.2f} kg")
print()

# Test different weights against this history
test_weights = [42.22, 50, 60, 70, 75, 80, 85, 90, 95, 100]
print("Plausibility scores for different weights:")
print("-" * 40)
for w in test_weights:
    score = scorer.calculate_plausibility_score(w, recent_weights)
    z_score = abs(w - np.mean(recent_weights)) / max(np.std(recent_weights), 0.5)
    print(f"{w:6.2f} kg: score={score:.3f}, z-score={z_score:.2f}")

print("\n" + "=" * 60)
print("Testing consistency with shorter time gaps:")
print("-" * 40)

# Test 42.22kg with different time gaps
previous_weight = 88.0
for days in [1, 7, 30, 60, 90, 180, 237]:
    hours = days * 24
    consistency = scorer.calculate_consistency_score(
        weight=42.22,
        previous_weight=previous_weight,
        time_diff_hours=hours
    )
    daily_rate = abs(42.22 - previous_weight) / days
    print(f"After {days:3d} days: consistency={consistency:.3f}, daily_rate={daily_rate:.2f} kg/day")

print("\n" + "=" * 60)
print("Analyzing the consistency scoring logic:")
print("-" * 40)

from constants import PHYSIOLOGICAL_LIMITS

max_daily = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
typical_daily = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']

print(f"MAX_DAILY_CHANGE_KG: {max_daily} kg")
print(f"TYPICAL_DAILY_VARIATION_KG: {typical_daily} kg")
print()

# Show how consistency score changes with daily rate
print("Daily rate -> Consistency score mapping:")
for rate in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.44, 7.0, 8.0, 10.0]:
    if rate <= typical_daily:
        score = 1.0
    elif rate <= max_daily:
        ratio = (rate - typical_daily) / (max_daily - typical_daily)
        score = 1.0 - (0.5 * ratio)
    else:
        excess_ratio = (rate - max_daily) / max_daily
        score = 0.5 * np.exp(-2 * excess_ratio)
    print(f"  {rate:5.2f} kg/day -> {score:.3f}")
