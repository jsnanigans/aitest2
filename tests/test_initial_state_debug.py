#!/usr/bin/env python3
"""Debug initial state calculation."""

import pandas as pd
import numpy as np

# Load data for specific user
df = pd.read_csv('data/2025-09-05_optimized.csv')
user_id = '0093a653-476b-4401-bbec-33a89abc2b18'
user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')

# Get first 10 measurements
first_10 = user_data.head(10)
weights = first_10['weight'].values.tolist()

print("Tracing the cleaning algorithm step by step:")
print(f"Initial weights: {weights}")
print()

# Clean weights for baseline (same logic as processor)
weights_clean = []
for i, w in enumerate(weights):
    print(f"Step {i+1}: Processing weight {w:.2f}")
    if not weights_clean:
        weights_clean.append(w)
        print(f"  First weight, adding to clean list: {weights_clean}")
    else:
        median = np.median(weights_clean)
        diff_pct = abs(w - median) / median
        print(f"  Current clean list: {weights_clean}")
        print(f"  Median of clean: {median:.2f}")
        print(f"  Difference from median: {diff_pct:.3f} ({diff_pct*100:.1f}%)")
        if diff_pct < 0.1:
            weights_clean.append(w)
            print(f"  Within 10%, adding to clean list")
        else:
            print(f"  >10% different, EXCLUDING from clean list")
    print()

print(f"Final clean weights: {weights_clean}")
print(f"Length of clean weights: {len(weights_clean)}")

if len(weights_clean) < 3:
    print("Less than 3 clean weights, using all weights")
    weights_clean = weights

baseline = np.median(weights_clean)
print(f"\nFinal baseline (median): {baseline:.2f} kg")
print(f"First measurement: {weights[0]:.2f} kg")
print(f"Baseline - First: {baseline - weights[0]:.2f} kg")

print("\n=== EXPLANATION ===")
print("The algorithm is working as designed:")
print("1. The first weight (114.76) starts the clean list")
print("2. The second weight (107.86) is 6% different from 114.76, so it's INCLUDED")
print("3. Once both are in, the median becomes 111.31")
print("4. Subsequent weights around 107 are within 10% of the growing median")
print("5. The median gradually shifts down as more 107-range values are added")
print("6. Final median of all 10 weights is 107.46 kg")
print("\nThis is actually reasonable - the algorithm adapts to the user's")
print("true weight range by using the median, which is robust to outliers.")