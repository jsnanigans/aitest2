import sys
sys.path.insert(0, 'src')
import numpy as np

def weighted_harmonic_mean(components, weights):
    """Calculate weighted harmonic mean."""
    total_weight = sum(weights.get(k, 0) for k in components)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = 0
    for key, score in components.items():
        weight = weights.get(key, 0)
        if score > 0:
            weighted_sum += weight / score
        else:
            return 0.0
    
    if weighted_sum == 0:
        return 0.0
    
    return total_weight / weighted_sum

def weighted_arithmetic_mean(components, weights):
    """Calculate weighted arithmetic mean."""
    total_weight = sum(weights.get(k, 0) for k in components)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(
        components.get(k, 0) * weights.get(k, 0)
        for k in components
    )
    
    return weighted_sum / total_weight

# Test cases with one low component
test_cases = [
    {"safety": 1.0, "plausibility": 0.3, "consistency": 1.0, "reliability": 0.77},
    {"safety": 1.0, "plausibility": 0.5, "consistency": 1.0, "reliability": 0.77},
    {"safety": 1.0, "plausibility": 0.7, "consistency": 1.0, "reliability": 0.77},
    {"safety": 1.0, "plausibility": 1.0, "consistency": 0.3, "reliability": 0.77},
    {"safety": 1.0, "plausibility": 1.0, "consistency": 0.5, "reliability": 0.77},
]

weights = {
    'safety': 0.35,
    'plausibility': 0.25,
    'consistency': 0.25,
    'reliability': 0.15
}

print("Comparing Harmonic vs Arithmetic Mean:")
print("=" * 70)
print("Components                                      Harmonic  Arithmetic  Diff")
print("-" * 70)

for components in test_cases:
    harmonic = weighted_harmonic_mean(components, weights)
    arithmetic = weighted_arithmetic_mean(components, weights)
    diff = arithmetic - harmonic
    
    comp_str = f"S={components['safety']:.1f} P={components['plausibility']:.1f} C={components['consistency']:.1f} R={components['reliability']:.2f}"
    print(f"{comp_str:45s}  {harmonic:.3f}     {arithmetic:.3f}      {diff:+.3f}")

print("\n" + "=" * 70)
print("Impact on acceptance (threshold = 0.6):")
print("-" * 70)

for components in test_cases:
    harmonic = weighted_harmonic_mean(components, weights)
    arithmetic = weighted_arithmetic_mean(components, weights)
    
    h_accept = "✓" if harmonic >= 0.6 else "✗"
    a_accept = "✓" if arithmetic >= 0.6 else "✗"
    
    comp_str = f"P={components['plausibility']:.1f} C={components['consistency']:.1f}"
    print(f"{comp_str:20s}  Harmonic: {harmonic:.3f} {h_accept}  Arithmetic: {arithmetic:.3f} {a_accept}")
