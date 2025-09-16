#!/usr/bin/env python3

import numpy as np
from typing import List, Optional, Tuple

def calculate_trend(weights: List[float]) -> Tuple[float, float]:
    """Calculate linear trend in weights.
    
    Returns:
        (slope, r_squared) where slope is kg/measurement and r_squared is fit quality
    """
    if len(weights) < 2:
        return 0.0, 0.0
    
    x = np.arange(len(weights))
    y = np.array(weights)
    
    # Linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, r_squared

def calculate_plausibility_with_trend(
    weight: float,
    recent_weights: Optional[List[float]],
    previous_weight: Optional[float]
) -> float:
    """Enhanced plausibility that accounts for trends."""
    
    # Use recent weights if available
    if recent_weights and len(recent_weights) >= 3:
        recent_array = np.array(recent_weights[-20:])
        mean = np.mean(recent_array)
        std = np.std(recent_array)
        
        # Calculate trend
        slope, r_squared = calculate_trend(list(recent_array))
        
        # If there's a clear trend (R² > 0.5), project it forward
        if r_squared > 0.5:
            # Project one step forward
            expected_next = recent_array[-1] + slope
            
            # Use projected value for mean if trend is strong
            if r_squared > 0.8:
                mean = expected_next
            else:
                # Blend projection with historical mean based on R²
                mean = (r_squared * expected_next) + ((1 - r_squared) * mean)
        
        # Adjust minimum std based on trend strength and consistency
        if abs(slope) > 0.1 and r_squared > 0.5:
            # For strong trends, allow more variation
            min_std = max(1.0, abs(slope) * 3)
        else:
            # Default minimum
            min_std = 0.5
        
        std = max(std, min_std)
        
    # Fall back to previous weight
    elif previous_weight is not None:
        mean = previous_weight
        # Assume 2% standard deviation for body weight
        baseline = (weight + previous_weight) / 2
        std = max(baseline * 0.02, 0.5)
        
    else:
        # No history available
        return 0.8
    
    # Calculate z-score
    z_score = abs(weight - mean) / std
    
    # Score based on deviation
    if z_score <= 1:
        return 1.0
    elif z_score <= 2:
        return 0.9
    elif z_score <= 3:
        return 0.7
    else:
        return max(0.0, min(0.5, np.exp(-0.5 * (z_score - 3))))

def test_enhanced_plausibility():
    print("=" * 80)
    print("TESTING ENHANCED PLAUSIBILITY WITH TREND DETECTION")
    print("=" * 80)
    
    # Gradual weight loss scenario
    weight_history = [82.0, 81.7, 81.4, 81.1, 80.87]
    test_weights = [80.5, 80.0, 79.0, 78.0, 77.0, 76.0]
    
    print("\nWeight history:", weight_history)
    slope, r_squared = calculate_trend(weight_history)
    print(f"Trend: {slope:.3f} kg/day, R² = {r_squared:.3f}")
    
    print("\n" + "-" * 60)
    print("ORIGINAL vs ENHANCED PLAUSIBILITY")
    print("-" * 60)
    
    # Import original scorer
    from src.quality_scorer import QualityScorer
    original_scorer = QualityScorer()
    
    for weight in test_weights:
        # Original calculation
        orig_score = original_scorer._calculate_plausibility(
            weight=weight,
            recent_weights=weight_history,
            previous_weight=weight_history[-1]
        )
        
        # Enhanced calculation
        enhanced_score = calculate_plausibility_with_trend(
            weight=weight,
            recent_weights=weight_history,
            previous_weight=weight_history[-1]
        )
        
        diff = weight - weight_history[-1]
        
        orig_status = "✓" if orig_score >= 0.7 else "⚠" if orig_score >= 0.5 else "✗"
        enh_status = "✓" if enhanced_score >= 0.7 else "⚠" if enhanced_score >= 0.5 else "✗"
        
        print(f"{weight:6.1f}kg (diff: {diff:+5.2f}kg) | "
              f"Original: {orig_status} {orig_score:.3f} | "
              f"Enhanced: {enh_status} {enhanced_score:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST WITH LONGER HISTORY (CLEAR TREND)")
    print("-" * 80)
    
    extended_history = [85.0, 84.5, 84.0, 83.5, 83.0, 82.5, 82.0, 81.7, 81.4, 81.1, 80.87]
    
    print("\nExtended history:", extended_history)
    slope, r_squared = calculate_trend(extended_history)
    print(f"Trend: {slope:.3f} kg/day, R² = {r_squared:.3f}")
    
    print("\n" + "-" * 60)
    
    for weight in test_weights:
        # Original calculation
        orig_score = original_scorer._calculate_plausibility(
            weight=weight,
            recent_weights=extended_history,
            previous_weight=extended_history[-1]
        )
        
        # Enhanced calculation
        enhanced_score = calculate_plausibility_with_trend(
            weight=weight,
            recent_weights=extended_history,
            previous_weight=extended_history[-1]
        )
        
        diff = weight - extended_history[-1]
        
        orig_status = "✓" if orig_score >= 0.7 else "⚠" if orig_score >= 0.5 else "✗"
        enh_status = "✓" if enhanced_score >= 0.7 else "⚠" if enhanced_score >= 0.5 else "✗"
        
        print(f"{weight:6.1f}kg (diff: {diff:+5.2f}kg) | "
              f"Original: {orig_status} {orig_score:.3f} | "
              f"Enhanced: {enh_status} {enhanced_score:.3f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print("The enhanced algorithm:")
    print("1. Detects linear trends in weight history")
    print("2. Projects the trend forward for expected next weight")
    print("3. Adjusts the minimum std deviation based on trend strength")
    print("4. Accepts legitimate weight changes that follow the trend")

if __name__ == "__main__":
    test_enhanced_plausibility()