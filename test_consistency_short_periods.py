import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer
from constants import PHYSIOLOGICAL_LIMITS
import numpy as np

scorer = QualityScorer()

print("Consistency scoring for SHORT time periods:")
print("=" * 70)
print(f"MAX_DAILY_CHANGE_KG: {PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']} kg")
print(f"TYPICAL_DAILY_VARIATION_KG: {PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']} kg")
print()

# Test small weight changes over short periods
print("Weight change -> Consistency score (for different time periods)")
print("-" * 70)

weight_changes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
time_periods = [1, 2, 6, 12, 24]  # hours

for change in weight_changes:
    print(f"\n{change:.1f} kg change:")
    for hours in time_periods:
        score = scorer.calculate_consistency_score(
            weight=90.0 + change,
            previous_weight=90.0,
            time_diff_hours=hours
        )
        daily_rate = (change / hours) * 24
        print(f"  {hours:2d}h (rate: {daily_rate:5.2f} kg/day) -> {score:.3f}")

print("\n" + "=" * 70)
print("The problem analysis:")
print("-" * 70)
print("For SHORT time periods (1-6 hours), even small weight changes")
print("can result in high 'daily rates' that trigger low consistency scores.")
print()
print("Example: 2kg change in 2 hours = 24 kg/day rate!")
print("This is unrealistic for human weight measurement patterns.")
print()
print("RECOMMENDATION: Adjust consistency scoring to be more lenient")
print("for short time periods, or use a minimum time threshold.")
