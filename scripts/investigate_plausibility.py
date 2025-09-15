#!/usr/bin/env python3

import numpy as np
from src.quality_scorer import QualityScorer

def test_plausibility_scenarios():
    scorer = QualityScorer()
    
    print("=" * 80)
    print("PLAUSIBILITY SCORE INVESTIGATION")
    print("Testing weight transitions from 80.87kg")
    print("=" * 80)
    
    previous_weight = 80.87
    test_weights = [76.0, 76.5, 77.0, 77.5, 78.0, 78.5, 79.0, 79.5, 80.0, 80.5, 81.0]
    
    print("\n1. WITHOUT RECENT HISTORY (uses previous_weight only)")
    print("-" * 60)
    
    for weight in test_weights:
        plausibility = scorer._calculate_plausibility(
            weight=weight,
            recent_weights=None,
            previous_weight=previous_weight
        )
        
        mean = previous_weight
        baseline = (weight + previous_weight) / 2
        std = max(baseline * 0.02, 0.5)
        z_score = abs(weight - mean) / std
        
        diff_kg = weight - previous_weight
        diff_percent = (abs(diff_kg) / previous_weight) * 100
        
        status = "✓" if plausibility >= 0.7 else "⚠" if plausibility >= 0.5 else "✗"
        
        print(f"{status} Weight: {weight:6.2f}kg | Diff: {diff_kg:+6.2f}kg ({diff_percent:5.2f}%) | "
              f"Z-score: {z_score:5.2f} | Plausibility: {plausibility:.3f} | "
              f"StdDev used: {std:.3f}kg")
    
    print("\n2. WITH RECENT HISTORY (simulating gradual weight loss)")
    print("-" * 60)
    
    recent_weights = [82.0, 81.8, 81.5, 81.2, 81.0, 80.87]
    print(f"Recent weights: {recent_weights}")
    
    for weight in test_weights:
        plausibility = scorer._calculate_plausibility(
            weight=weight,
            recent_weights=recent_weights,
            previous_weight=previous_weight
        )
        
        recent_array = np.array(recent_weights[-20:])
        mean = np.mean(recent_array)
        std = max(np.std(recent_array), 0.5)
        z_score = abs(weight - mean) / std
        
        diff_kg = weight - previous_weight
        diff_percent = (abs(diff_kg) / previous_weight) * 100
        
        status = "✓" if plausibility >= 0.7 else "⚠" if plausibility >= 0.5 else "✗"
        
        print(f"{status} Weight: {weight:6.2f}kg | Diff: {diff_kg:+6.2f}kg ({diff_percent:5.2f}%) | "
              f"Z-score: {z_score:5.2f} | Plausibility: {plausibility:.3f} | "
              f"Mean: {mean:.2f}kg, StdDev: {std:.3f}kg")
    
    print("\n3. DETAILED ANALYSIS FOR 78kg")
    print("-" * 60)
    
    weight = 78.0
    
    print("\nWithout history:")
    mean = previous_weight
    baseline = (weight + previous_weight) / 2
    std = max(baseline * 0.02, 0.5)
    z_score = abs(weight - mean) / std
    
    print(f"  Mean (previous weight): {mean:.2f}kg")
    print(f"  Baseline (avg of current & previous): {baseline:.2f}kg")
    print(f"  StdDev (2% of baseline, min 0.5kg): {std:.3f}kg")
    print(f"  Z-score: |{weight} - {mean:.2f}| / {std:.3f} = {z_score:.3f}")
    
    if z_score <= 1:
        score = 1.0
        print(f"  Z-score ≤ 1: Score = 1.0")
    elif z_score <= 2:
        score = 0.9
        print(f"  1 < Z-score ≤ 2: Score = 0.9")
    elif z_score <= 3:
        score = 0.7
        print(f"  2 < Z-score ≤ 3: Score = 0.7")
    else:
        score = max(0.0, min(0.5, np.exp(-0.5 * (z_score - 3))))
        print(f"  Z-score > 3: Score = max(0.0, min(0.5, exp(-0.5 * ({z_score:.3f} - 3)))) = {score:.3f}")
    
    print(f"  Final plausibility score: {score:.3f}")
    
    print("\n4. OVERALL QUALITY SCORES")
    print("-" * 60)
    
    for weight in [76.0, 77.0, 78.0, 79.0, 80.0]:
        quality = scorer.calculate_quality_score(
            weight=weight,
            source='patient-upload',
            previous_weight=previous_weight,
            time_diff_hours=24.0,
            recent_weights=None,
            user_height_m=1.70
        )
        
        status = "✓ ACCEPTED" if quality.accepted else "✗ REJECTED"
        print(f"\nWeight: {weight}kg | Overall: {quality.overall:.3f} | {status}")
        print(f"  Components: Safety={quality.components['safety']:.3f}, "
              f"Plausibility={quality.components['plausibility']:.3f}, "
              f"Consistency={quality.components['consistency']:.3f}, "
              f"Reliability={quality.components['reliability']:.3f}")
        if quality.rejection_reason:
            print(f"  Reason: {quality.rejection_reason}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)
    print("1. Plausibility uses 2% standard deviation when no history available")
    print("2. For 80.87kg → 78kg: z-score = 2.87/1.59 = 1.81 → score = 0.9")
    print("3. This is reasonable for gradual weight loss")
    print("4. The issue might be with the overall score calculation or other components")

if __name__ == "__main__":
    test_plausibility_scenarios()