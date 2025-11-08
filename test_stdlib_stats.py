#!/usr/bin/env python3
"""
Test numpy statistical functions and compare with TypeScript
"""

import json
import numpy as np

def main():
    print("=== Testing numpy Statistical Functions ===\n")

    test_cases = [
        {
            "name": "Simple sequence",
            "data": [1, 2, 3, 4, 5],
        },
        {
            "name": "Real weight data (kg)",
            "data": [104.3, 104.3, 113.4, 117.9, 115.4],
        },
        {
            "name": "Single value",
            "data": [42.0],
        },
        {
            "name": "Two values",
            "data": [10.0, 20.0],
        },
        {
            "name": "With negative values",
            "data": [-5, 0, 5, 10],
        },
        {
            "name": "Large variance",
            "data": [1, 100, 1, 100, 1],
        },
        {
            "name": "Disputed measurement context",
            "data": [104.3, 104.3],
        },
    ]

    results = []

    for tc in test_cases:
        data = np.array(tc["data"])

        mean_val = float(np.mean(data))
        std_val = float(np.std(data, ddof=1))  # Sample std dev
        var_val = float(np.var(data, ddof=1))  # Sample variance
        median_val = float(np.median(data))

        result = {
            "data": tc["data"],
            "mean": mean_val,
            "std": std_val,
            "variance": var_val,
            "description": tc["name"],
        }

        results.append(result)

        print(f"\n{tc['name']}:")
        print(f"  Data: [{', '.join(map(str, tc['data']))}]")
        print(f"  Mean: {mean_val:.15f}")
        print(f"  Std Dev (sample): {std_val:.15f}")
        print(f"  Variance (sample): {var_val:.15f}")
        print(f"  Median: {median_val:.15f}")

    # Write results
    with open("py_stats_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results written to py_stats_results.json")

    # Compare with TypeScript
    try:
        with open("ts_stats_results.json", "r") as f:
            ts_results = json.load(f)

        print("\n=== Comparison with TypeScript ===\n")

        all_match = True

        for i, (ts, py) in enumerate(zip(ts_results, results)):
            print(f"\n{ts['description']}:")

            mean_diff = abs(ts["mean"] - py["mean"])
            std_diff = abs(ts["std"] - py["std"])
            var_diff = abs(ts["variance"] - py["variance"])

            mean_match = mean_diff < 1e-14
            std_match = std_diff < 1e-14
            var_match = var_diff < 1e-14

            print(f"  Mean: TS={ts['mean']:.15f} Py={py['mean']:.15f} diff={mean_diff:.2e} {'✓' if mean_match else '✗'}")
            print(f"  Std:  TS={ts['std']:.15f} Py={py['std']:.15f} diff={std_diff:.2e} {'✓' if std_match else '✗'}")
            print(f"  Var:  TS={ts['variance']:.15f} Py={py['variance']:.15f} diff={var_diff:.2e} {'✓' if var_match else '✗'}")

            if not (mean_match and std_match and var_match):
                all_match = False

        print("\n=== Summary ===")
        if all_match:
            print("✅ All statistical functions match to machine precision!")
        else:
            print("❌ Statistical functions have differences!")

    except FileNotFoundError:
        print("\n⚠️  TypeScript results not found. Run test_stdlib_stats.ts first.")

if __name__ == "__main__":
    main()
