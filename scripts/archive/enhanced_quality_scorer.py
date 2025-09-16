"""
Enhanced quality scoring incorporating physiological insights from research.
Based on findings that body weight naturally fluctuates 1-2 kg daily (2-3% of body weight).
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
from datetime import datetime
from constants import PHYSIOLOGICAL_LIMITS

def calculate_consistency_score_enhanced(
    weight: float,
    previous_weight: float = None,
    time_diff_hours: float = None,
    user_baseline_weight: float = None
) -> float:
    """
    Enhanced consistency scoring based on physiological research.
    
    Key insights:
    - Daily fluctuations of 1-2 kg (2-3% of body weight) are normal
    - Weekly patterns show 0.35% variation
    - Thresholds should be proportional to baseline weight
    """
    if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
        return 0.8
    
    # Use baseline weight for percentage calculations
    # If not available, use average of current and previous
    if user_baseline_weight is None:
        user_baseline_weight = (weight + previous_weight) / 2
    
    weight_diff = abs(weight - previous_weight)
    weight_diff_percent = (weight_diff / user_baseline_weight) * 100
    
    # Define physiologically-based thresholds
    if time_diff_hours < 6:
        # Within 6 hours: up to 2% change is normal (hydration, meals)
        typical_percent = 1.0
        max_percent = 2.0
        
    elif time_diff_hours < 24:
        # Within a day: 2-3% change is documented as normal
        typical_percent = 1.5
        max_percent = 3.0
        
    elif time_diff_hours < 168:  # Within a week
        # Weekly variation ~0.35% per day on average
        days = time_diff_hours / 24
        typical_percent = 0.35 * days
        max_percent = 0.5 * days + 2.0  # Allow for weekend effect
        
    else:
        # Longer term: use conservative daily rate
        days = time_diff_hours / 24
        typical_percent = 0.2 * days  # ~0.2% per day for gradual changes
        max_percent = 0.35 * days  # ~0.35% per day maximum
    
    # Calculate score based on percentage change
    if weight_diff_percent <= typical_percent:
        return 1.0
    elif weight_diff_percent <= max_percent:
        ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
        return 1.0 - (0.4 * ratio)
    else:
        excess_ratio = (weight_diff_percent - max_percent) / max_percent
        return 0.6 * np.exp(-2 * excess_ratio)

# Test the enhanced scoring
print("Enhanced Consistency Scoring (Physiologically-Informed)")
print("=" * 70)

# Test with different baseline weights
baseline_weights = [70, 90, 120]  # kg
weight_changes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # kg
time_periods = [1, 6, 24, 168]  # hours (1h, 6h, 1 day, 1 week)

for baseline in baseline_weights:
    print(f"\nBaseline weight: {baseline} kg")
    print("-" * 60)
    
    for change in weight_changes:
        change_percent = (change / baseline) * 100
        print(f"\n{change:.1f} kg change ({change_percent:.1f}% of body weight):")
        
        for hours in time_periods:
            score = calculate_consistency_score_enhanced(
                weight=baseline + change,
                previous_weight=baseline,
                time_diff_hours=hours,
                user_baseline_weight=baseline
            )
            
            time_label = f"{hours}h" if hours < 24 else f"{hours/24:.0f}d"
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            print(f"  {time_label:4s}: {score:.3f} {status}")

print("\n" + "=" * 70)
print("Key Improvements:")
print("-" * 60)
print("1. Thresholds scale with body weight (percentage-based)")
print("2. Incorporates 2-3% daily fluctuation as normal")
print("3. Accounts for weekly patterns (0.35% variation)")
print("4. More physiologically accurate for all body types")
