import sys
sys.path.insert(0, 'src')
import numpy as np
from constants import PHYSIOLOGICAL_LIMITS

def calculate_consistency_score_improved(
    weight: float,
    previous_weight: float = None,
    time_diff_hours: float = None
) -> float:
    """
    Improved consistency scoring that handles short time periods better.
    """
    if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
        return 0.8
    
    weight_diff = abs(weight - previous_weight)
    
    # Key improvement: Use different thresholds based on time period
    if time_diff_hours < 6:
        # For measurements within 6 hours, allow up to 3kg variation
        # This accounts for normal daily fluctuations, hydration, meals, etc.
        max_allowed = 3.0
        typical_allowed = 1.5
        
        if weight_diff <= typical_allowed:
            return 1.0
        elif weight_diff <= max_allowed:
            ratio = (weight_diff - typical_allowed) / (max_allowed - typical_allowed)
            return 1.0 - (0.3 * ratio)  # Less harsh penalty
        else:
            excess_ratio = (weight_diff - max_allowed) / max_allowed
            return 0.7 * np.exp(-2 * excess_ratio)
            
    elif time_diff_hours < 24:
        # For measurements within a day, use scaled thresholds
        # Interpolate between short-term and daily limits
        hours_ratio = time_diff_hours / 24
        max_allowed = 3.0 + (PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG'] - 3.0) * hours_ratio
        typical_allowed = 1.5 + (PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG'] - 1.5) * hours_ratio
        
        if weight_diff <= typical_allowed:
            return 1.0
        elif weight_diff <= max_allowed:
            ratio = (weight_diff - typical_allowed) / (max_allowed - typical_allowed)
            return 1.0 - (0.4 * ratio)
        else:
            excess_ratio = (weight_diff - max_allowed) / max_allowed
            return 0.6 * np.exp(-2 * excess_ratio)
    
    else:
        # For longer periods, use daily rate as before
        daily_rate = (weight_diff / time_diff_hours) * 24
        max_daily = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
        typical_daily = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']
        
        if daily_rate <= typical_daily:
            return 1.0
        elif daily_rate <= max_daily:
            ratio = (daily_rate - typical_daily) / (max_daily - typical_daily)
            return 1.0 - (0.5 * ratio)
        else:
            excess_ratio = (daily_rate - max_daily) / max_daily
            return 0.5 * np.exp(-2 * excess_ratio)

def calculate_consistency_score_original(weight, previous_weight, time_diff_hours):
    """Original consistency scoring for comparison."""
    if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
        return 0.8
    
    weight_diff = abs(weight - previous_weight)
    daily_rate = (weight_diff / time_diff_hours) * 24
    
    max_daily = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
    typical_daily = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']
    
    if daily_rate <= typical_daily:
        return 1.0
    elif daily_rate <= max_daily:
        ratio = (daily_rate - typical_daily) / (max_daily - typical_daily)
        return 1.0 - (0.5 * ratio)
    else:
        excess_ratio = (daily_rate - max_daily) / max_daily
        return 0.5 * np.exp(-2 * excess_ratio)

# Test the improved function
print("Improved Consistency Scoring:")
print("=" * 70)

weight_changes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
time_periods = [1, 2, 6, 12, 24]

for change in weight_changes:
    print(f"\n{change:.1f} kg change:")
    print("  Hours:  Original  Improved  Difference")
    print("  " + "-" * 40)
    for hours in time_periods:
        original_score = calculate_consistency_score_original(90.0 + change, 90.0, hours)
        improved_score = calculate_consistency_score_improved(90.0 + change, 90.0, hours)
        diff = improved_score - original_score
        symbol = "↑" if diff > 0.1 else "↓" if diff < -0.1 else "≈"
        print(f"  {hours:3d}h:    {original_score:.3f}     {improved_score:.3f}      {diff:+.3f} {symbol}")
