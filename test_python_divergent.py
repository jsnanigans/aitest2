#!/usr/bin/env python3
"""Check Python's quality scores for divergent measurements."""

import sys
import json
from pathlib import Path

# Read the Python results
results_file = Path("output_comparison_py").glob("local_processing_results_*.json")
results_file = next(results_file, None)

if not results_file:
    print("No Python results file found!")
    sys.exit(1)

with open(results_file) as f:
    results = json.load(f)

# IDs that diverge
divergent_ids = [
    "726b441f-eb43-47d9-8f3c-845d164e5a5b",  # TS accepts, Python should reject
    "86233705-0332-44f1-bc69-bc796220f598",  # Python accepts, TS rejects
    "1a98b2c3-e023-4757-8d01-d35ef2fb363e",  # Python accepts, TS rejects
    "510977fa-9d3f-4b50-a667-e676a0cc0791",  # Python accepts, TS rejects
]

print("\n=== Python Quality Scores for Divergent Measurements ===\n")

for user_id, user_results in results.items():
    for result in user_results:
        measurement_id = result.get("measurement_id")
        if measurement_id in divergent_ids:
            print(f"Measurement: {measurement_id[:8]}...")
            print(f"  Weight: {result.get('weight')} {result.get('unit')}")
            print(f"  Quality Score: {result.get('quality_score', 0):.15f}")
            print(f"  Accepted: {result.get('accepted')}")

            if "quality_components" in result:
                print(f"\n  Component Scores:")
                components = result["quality_components"]
                weights = {
                    "kalman_fit": 0.30,
                    "temporal_consistency": 0.25,
                    "anomaly_detection": 0.25,
                    "source_reliability": 0.10,
                    "trend_alignment": 0.10,
                }
                total = 0
                for name, score in components.items():
                    weight = weights.get(name, 0)
                    contrib = score * weight
                    total += contrib
                    print(f"    {name:25} {score:.6f} × {weight:.2f} = {contrib:.6f}")
                print(f"    {'TOTAL':25} {' ' * 9} {' ' * 6} {total:.6f}")
            print()
