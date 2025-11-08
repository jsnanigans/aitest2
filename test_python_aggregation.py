#!/usr/bin/env python3
"""
Test which aggregation method Python actually uses
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python_lib" / "src"))

from weight_processor_lib.core.processing.unified_quality_scorer import UnifiedQualityScorer

# Component scores for measurement 726b441f
components = {
    "kalman_fit": 0.867847,
    "temporal_consistency": 0.361080,
    "anomaly_detection": 0.714157,
    "source_reliability": 0.800000,
    "trend_alignment": 0.616598,
}

weights = {
    "kalman_fit": {"weight": 0.30, "enabled": True},
    "temporal_consistency": {"weight": 0.25, "enabled": True},
    "anomaly_detection": {"weight": 0.25, "enabled": True},
    "source_reliability": {"weight": 0.10, "enabled": True},
    "trend_alignment": {"weight": 0.10, "enabled": True},
}

# Test weighted sum (arithmetic)
weighted_sum = sum(score * weights[name]["weight"] for name, score in components.items())
print("Weighted Sum (Arithmetic):", f"{weighted_sum:.15f}")

# Test geometric mean
product = 1.0
weight_sum = 0.0
for name, score in components.items():
    weight = weights[name]["weight"]
    product *= score ** weight
    weight_sum += weight

geometric_mean = product ** (1.0 / weight_sum)
print("Geometric Mean:          ", f"{geometric_mean:.15f}")

# Create scorer and test
config = {
    "component_weights": {  # Changed from "components" to "component_weights"
        "kalman_fit": 0.30,
        "temporal_consistency": 0.25,
        "anomaly_detection": 0.25,
        "source_reliability": 0.10,
        "trend_alignment": 0.10,
    },
    "threshold": 0.5,
}
scorer = UnifiedQualityScorer(config)
print("\n=== Checking Python Config ===")
print(f"Scorer weights: {scorer.weights}")
# Check if it's using harmonic or geometric
print("\nScorer config:")
print(f"  use_harmonic_mean: {getattr(scorer, 'use_harmonic_mean', False)}")

# Test the actual calculation
result = scorer._calculate_weighted_geometric_mean(components)
print(f"\nScorer geometric mean:   {result:.15f}")

result2 = scorer._calculate_weighted_harmonic_mean(components)
print(f"Scorer harmonic mean:    {result2:.15f}")

# Debug the scorer calculation step by step
print("\n=== Debugging Scorer Calculation ===")
epsilon = 1e-10
product = 1.0
weight_sum = 0.0

for component_name, score in components.items():
    weight = scorer.weights.get(component_name, 0.0)
    if weight > 0:
        # Clamp score to avoid numerical issues
        clamped = max(epsilon, min(1.0, score))
        contrib = clamped ** weight
        product *= contrib
        weight_sum += weight
        print(f"  {component_name.ljust(25)}: {score:.6f} clamped to {clamped:.6f}, weight {weight:.2f}, contrib {contrib:.15f}")

if weight_sum > 0:
    # Normalize by weight sum
    overall = product ** (1.0 / weight_sum)
    print(f"\nProduct: {product:.15f}")
    print(f"Weight sum: {weight_sum}")
    print(f"Overall (product ** (1/{weight_sum})): {overall:.15f}")
    print(f"Final clamped: {max(0.0, min(1.0, overall)):.15f}")
