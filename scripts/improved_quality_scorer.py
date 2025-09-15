"""
Improved Quality Scorer - Stateless Version
Designed to work with the stateless processor architecture where:
- Each measurement is processed independently
- State is loaded from database, used, then saved
- No instance variables or persistent memory
"""

import sys
sys.path.insert(0, 'src')
import numpy as np
from typing import Optional, List, Dict
from constants import PHYSIOLOGICAL_LIMITS

def calculate_consistency_score_improved(
    weight: float,
    previous_weight: Optional[float] = None,
    time_diff_hours: Optional[float] = None,
    recent_weights: Optional[List[float]] = None
) -> float:
    """
    Improved consistency scoring that works in stateless architecture.
    
    Key improvements:
    - Time-aware thresholds for short periods
    - Percentage-based scaling using recent_weights for context
    - No persistent state required
    
    Args:
        weight: Current measurement
        previous_weight: Last accepted weight from state
        time_diff_hours: Hours since last measurement
        recent_weights: Recent accepted weights from state (for baseline estimation)
    
    Returns:
        Consistency score from 0.0 to 1.0
    """
    if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
        return 0.8
    
    # Estimate baseline from recent history (stateless - computed each time)
    if recent_weights and len(recent_weights) >= 3:
        baseline_weight = np.median(recent_weights[-10:])
    else:
        baseline_weight = (weight + previous_weight) / 2
    
    weight_diff = abs(weight - previous_weight)
    weight_diff_percent = (weight_diff / baseline_weight) * 100
    
    # Time-aware thresholds based on research
    if time_diff_hours < 6:
        # Within 6 hours: 1.5% typical, 2.5% max
        typical_percent = 1.0
        max_percent = 2.5
        
        if weight_diff_percent <= typical_percent:
            return 1.0
        elif weight_diff_percent <= max_percent:
            ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
            return 1.0 - (0.3 * ratio)
        else:
            excess = (weight_diff_percent - max_percent) / max_percent
            return 0.7 * np.exp(-2 * excess)
            
    elif time_diff_hours < 24:
        # Within a day: scale between hourly and daily limits
        hours_ratio = time_diff_hours / 24
        
        # Interpolate thresholds
        typical_percent = 1.0 + (2.0 - 1.0) * hours_ratio
        max_percent = 2.5 + (3.5 - 2.5) * hours_ratio
        
        if weight_diff_percent <= typical_percent:
            return 1.0
        elif weight_diff_percent <= max_percent:
            ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
            return 1.0 - (0.4 * ratio)
        else:
            excess = (weight_diff_percent - max_percent) / max_percent
            return 0.6 * np.exp(-2 * excess)
    
    else:
        # For longer periods: use daily rate
        daily_rate_percent = weight_diff_percent / (time_diff_hours / 24)
        
        # Research shows 0.35% daily variation is normal for weekly patterns
        if daily_rate_percent <= 0.35:
            return 1.0
        elif daily_rate_percent <= 0.7:  # Double the normal rate
            ratio = (daily_rate_percent - 0.35) / 0.35
            return 1.0 - (0.5 * ratio)
        else:
            excess = (daily_rate_percent - 0.7) / 0.7
            return 0.5 * np.exp(-2 * excess)

def calculate_plausibility_score_improved(
    weight: float,
    recent_weights: Optional[List[float]] = None
) -> float:
    """
    Improved plausibility scoring using MAD for robustness.
    Stateless - all context comes from recent_weights parameter.
    
    Args:
        weight: Current measurement
        recent_weights: Recent accepted weights from state
    
    Returns:
        Plausibility score from 0.0 to 1.0
    """
    if not recent_weights or len(recent_weights) < 3:
        return 0.8
    
    # Use only recent history (stateless window)
    recent_array = np.array(recent_weights[-20:])
    median_weight = np.median(recent_array)
    
    # Use MAD for robust variance estimation
    mad = np.median(np.abs(recent_array - median_weight))
    
    # Handle zero MAD
    if mad < 0.1:
        mad = 0.5
    
    # Scale factor for normal distribution
    robust_std = 1.4826 * mad
    
    # Calculate robust z-score
    deviation = abs(weight - median_weight)
    z_score = deviation / robust_std
    
    # Percentage-based check as well
    deviation_percent = (deviation / median_weight) * 100
    
    # Combine z-score and percentage approaches
    if deviation_percent <= 2.0 and z_score <= 1.5:
        return 1.0
    elif deviation_percent <= 3.0 and z_score <= 2.5:
        return 0.9
    elif deviation_percent <= 5.0 and z_score <= 3.5:
        return 0.7
    else:
        # Exponential decay for extreme values
        score = np.exp(-0.5 * (z_score - 3))
        return max(0.0, min(0.5, score))

# Test the improved stateless functions
print("STATELESS QUALITY SCORING IMPROVEMENTS")
print("=" * 70)

# Simulate state that would be loaded from database
state_recent_weights = [92.5, 92.8, 93.0, 92.7, 92.9, 93.1, 92.6, 92.8]
state_previous_weight = 92.8

print("Simulated State (from database):")
print(f"  Previous weight: {state_previous_weight} kg")
print(f"  Recent weights: {state_recent_weights}")
print(f"  Baseline (median): {np.median(state_recent_weights):.2f} kg")
print()

# Test cases - each processed independently (stateless)
test_cases = [
    (92.5, 2, "Small decrease, 2 hours"),
    (93.5, 6, "Moderate increase, 6 hours"),
    (94.0, 24, "Larger increase, 1 day"),
    (91.0, 48, "Decrease, 2 days"),
    (42.22, 56, "Outlier"),
    (95.0, 168, "Increase, 1 week"),
]

print("Stateless Processing Results:")
print("-" * 70)

for weight, hours, description in test_cases:
    # Each call is completely independent - stateless
    consistency = calculate_consistency_score_improved(
        weight=weight,
        previous_weight=state_previous_weight,
        time_diff_hours=hours,
        recent_weights=state_recent_weights
    )
    
    plausibility = calculate_plausibility_score_improved(
        weight=weight,
        recent_weights=state_recent_weights
    )
    
    # Simple acceptance logic (would be in processor)
    # Using harmonic mean of the two scores
    if consistency > 0 and plausibility > 0:
        combined = 2 / (1/consistency + 1/plausibility)
    else:
        combined = 0.0
    
    accepted = combined >= 0.5
    
    print(f"\n{description}:")
    print(f"  Weight: {weight} kg")
    print(f"  Consistency: {consistency:.3f}")
    print(f"  Plausibility: {plausibility:.3f}")
    print(f"  Combined: {combined:.3f}")
    print(f"  Status: {'✓ ACCEPT' if accepted else '✗ REJECT'}")

print("\n" + "=" * 70)
print("KEY DESIGN PRINCIPLES FOR STATELESS ARCHITECTURE:")
print("-" * 70)
print("1. No instance variables - everything passed as parameters")
print("2. State loaded from database at start of processing")
print("3. Recent weights provide context for baseline estimation")
print("4. Each measurement processed independently")
print("5. State saved back to database after processing")
