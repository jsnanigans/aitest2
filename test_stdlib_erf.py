#!/usr/bin/env python3
"""
Test scipy.special.erf against same inputs as TypeScript
Compare results to find any differences
"""

import json
from scipy.special import erf
import numpy as np

def main():
    print("=== Testing scipy.special.erf function ===\n")

    # Same test inputs as TypeScript
    test_inputs = [
        0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        -0.1, -0.5, -1.0, -2.0,
        0.02, -0.02, 5.0, -5.0,
        0.6745, 1.96, 2.576,
    ]

    results = []

    print("Input\t\tOutput (erf)\t\tPrecision")
    print("=" * 60)

    for x in test_inputs:
        result = float(erf(x))
        results.append({
            "input": x,
            "output": result,
            "description": f"erf({x})",
        })

        print(f"{x:.6f}\t{result:.15f}\t\t15 digits")

    # Write results
    with open("py_erf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results written to py_erf_results.json")

    # Load and compare with TypeScript results
    try:
        with open("ts_erf_results.json", "r") as f:
            ts_results = json.load(f)

        print("\n=== Comparison with TypeScript ===\n")
        print("Input\t\tTS erf\t\t\tPy erf\t\t\tDifference")
        print("=" * 80)

        max_diff = 0.0
        max_diff_input = 0.0

        for i, (ts, py) in enumerate(zip(ts_results, results)):
            diff = abs(ts["output"] - py["output"])
            max_diff = max(max_diff, diff)
            if diff == max_diff:
                max_diff_input = ts["input"]

            match = "✓" if diff < 1e-14 else "✗"
            print(f"{ts['input']:.6f}\t{ts['output']:.15f}\t{py['output']:.15f}\t{diff:.2e} {match}")

        print("\n=== Summary ===")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"At input value: {max_diff_input}")

        if max_diff < 1e-14:
            print("✅ erf() functions match to machine precision!")
        else:
            print(f"❌ erf() functions differ by {max_diff:.2e}")

    except FileNotFoundError:
        print("\n⚠️  TypeScript results not found. Run test_stdlib_erf.ts first.")

if __name__ == "__main__":
    main()
