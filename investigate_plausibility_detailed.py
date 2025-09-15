#!/usr/bin/env python3

import numpy as np
from src.quality_scorer import QualityScorer

def test_real_scenario():
    scorer = QualityScorer()
    
    print("=" * 80)
    print("REAL SCENARIO: GRADUAL WEIGHT LOSS")
    print("=" * 80)
    
    print("\nSimulating a user losing weight gradually over time")
    print("Starting at 82kg, losing ~0.2-0.3kg per day")
    
    weight_history = [
        82.0,   # Day 1
        81.7,   # Day 2  
        81.4,   # Day 3
        81.1,   # Day 4
        80.87,  # Day 5 (current accepted weight)
    ]
    
    next_measurements = [
        80.5,   # Day 6 - reasonable continuation
        80.2,   # Day 6 alternative
        79.8,   # Day 6 alternative  
        79.0,   # Day 7 - faster loss
        78.0,   # Day 8 - user reports this is being rejected
        77.0,   # Day 9 - also rejected?
    ]
    
    print(f"\nWeight history: {weight_history}")
    print(f"Last accepted: {weight_history[-1]}kg")
    
    print("\n" + "=" * 80)
    print("TEST 1: Using only previous weight (no history)")
    print("-" * 80)
    
    for weight in next_measurements:
        quality = scorer.calculate_quality_score(
            weight=weight,
            source='patient-upload',
            previous_weight=weight_history[-1],
            time_diff_hours=24.0,
            recent_weights=None,
            user_height_m=1.70
        )
        
        diff = weight - weight_history[-1]
        status = "✓ ACCEPTED" if quality.accepted else "✗ REJECTED"
        
        print(f"\n{weight}kg (diff: {diff:+.2f}kg) → {status}")
        print(f"  Overall: {quality.overall:.3f} (threshold: {quality.threshold})")
        for comp, score in quality.components.items():
            marker = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            print(f"    {marker} {comp:12s}: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Using recent weight history")
    print("-" * 80)
    
    for weight in next_measurements:
        quality = scorer.calculate_quality_score(
            weight=weight,
            source='patient-upload',
            previous_weight=weight_history[-1],
            time_diff_hours=24.0,
            recent_weights=weight_history,
            user_height_m=1.70
        )
        
        diff = weight - weight_history[-1]
        status = "✓ ACCEPTED" if quality.accepted else "✗ REJECTED"
        
        print(f"\n{weight}kg (diff: {diff:+.2f}kg) → {status}")
        print(f"  Overall: {quality.overall:.3f} (threshold: {quality.threshold})")
        for comp, score in quality.components.items():
            marker = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            print(f"    {marker} {comp:12s}: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST 3: Using longer history with clear downward trend")
    print("-" * 80)
    
    extended_history = [85.0, 84.5, 84.0, 83.5, 83.0, 82.5] + weight_history
    print(f"Extended history: {extended_history}")
    
    for weight in [80.0, 79.0, 78.0, 77.0, 76.0]:
        quality = scorer.calculate_quality_score(
            weight=weight,
            source='patient-upload',
            previous_weight=weight_history[-1],
            time_diff_hours=24.0,
            recent_weights=extended_history,
            user_height_m=1.70
        )
        
        diff = weight - weight_history[-1]
        
        recent_array = np.array(extended_history[-20:])
        mean = np.mean(recent_array)
        std = max(np.std(recent_array), 0.5)
        z_score = abs(weight - mean) / std
        
        status = "✓ ACCEPTED" if quality.accepted else "✗ REJECTED"
        
        print(f"\n{weight}kg (diff: {diff:+.2f}kg) → {status}")
        print(f"  History mean: {mean:.2f}kg, std: {std:.3f}kg, z-score: {z_score:.2f}")
        print(f"  Overall: {quality.overall:.3f} (threshold: {quality.threshold})")
        for comp, score in quality.components.items():
            marker = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            print(f"    {marker} {comp:12s}: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("PROBLEM IDENTIFIED:")
    print("-" * 80)
    print("When using recent history with low variance (std = 0.5kg minimum),")
    print("legitimate weight loss measurements get extreme z-scores!")
    print("")
    print("For a consistent downward trend, the std of recent weights is small,")
    print("but the minimum std of 0.5kg makes ANY weight > 1.5kg from mean problematic.")
    print("")
    print("SOLUTION: The minimum std should be higher for gradual weight changes,")
    print("or the plausibility calculation should account for trends, not just variance.")

if __name__ == "__main__":
    test_real_scenario()