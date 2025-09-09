#!/usr/bin/env python3
"""Test initial state calculation for specific user."""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import WeightProcessor
from processor_database import ProcessorDatabase
import toml

# Load config
config = toml.load('config.toml')

# Load data for specific user
df = pd.read_csv('data/2025-09-05_optimized.csv')
user_id = '0093a653-476b-4401-bbec-33a89abc2b18'
user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')

print(f"User {user_id}")
print(f"Total measurements: {len(user_data)}")
print("\nFirst 10 measurements (used for initialization):")

# Get first 10 measurements
first_10 = user_data.head(10)
for idx, row in first_10.iterrows():
    print(f"  {row['effectivDateTime']}: {row['weight']:.2f} kg")

# Calculate what the initial state would be
weights = first_10['weight'].values.tolist()

# Clean weights for baseline (same logic as processor)
weights_clean = []
for w in weights:
    if not weights_clean:
        weights_clean.append(w)
    else:
        median = np.median(weights_clean)
        if abs(w - median) / median < 0.1:  # Within 10% of median
            weights_clean.append(w)

if len(weights_clean) < 3:
    weights_clean = weights

baseline = np.median(weights_clean)

print(f"\nInitialization analysis:")
print(f"  All weights: {[f'{w:.2f}' for w in weights]}")
print(f"  Clean weights (within 10% of running median): {[f'{w:.2f}' for w in weights_clean]}")
print(f"  Baseline (median of clean): {baseline:.2f} kg")
print(f"  First measurement: {weights[0]:.2f} kg")
print(f"  Difference: {baseline - weights[0]:.2f} kg")

# Check for other users with similar pattern
print("\n\nChecking other users with similar patterns...")

# Process all users
users_with_high_init = []
for uid in df['user_id'].unique():
    user_df = df[df['user_id'] == uid].sort_values('effectivDateTime')
    if len(user_df) >= 10:
        first_10_weights = user_df.head(10)['weight'].values
        
        # Same cleaning logic
        weights_clean = []
        for w in first_10_weights:
            if not weights_clean:
                weights_clean.append(w)
            else:
                median = np.median(weights_clean)
                if abs(w - median) / median < 0.1:
                    weights_clean.append(w)
        
        if len(weights_clean) < 3:
            weights_clean = first_10_weights
            
        baseline = np.median(weights_clean)
        first = first_10_weights[0]
        
        if baseline > first + 2:  # Initial state > 2kg higher than first
            users_with_high_init.append({
                'user_id': uid,
                'first': first,
                'baseline': baseline,
                'diff': baseline - first,
                'first_10': first_10_weights
            })

print(f"Found {len(users_with_high_init)} users where initial state > first measurement + 2kg")
print("\nTop 5 cases:")
sorted_users = sorted(users_with_high_init, key=lambda x: x['diff'], reverse=True)[:5]
for u in sorted_users:
    print(f"\nUser {u['user_id']}:")
    print(f"  First: {u['first']:.2f} kg")
    print(f"  Initial state: {u['baseline']:.2f} kg") 
    print(f"  Difference: {u['diff']:.2f} kg")
    print(f"  First 10: {[f'{w:.2f}' for w in u['first_10']]}")