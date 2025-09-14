import sys
sys.path.insert(0, 'src')
import csv
from datetime import datetime
import numpy as np

# Read all data for this user
user_id = "03de147f-5e59-49b5-864b-da235f1dab54"
measurements = []

with open('data/2025-09-05_optimized.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == user_id:
            measurements.append({
                'timestamp': datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'),
                'source': row[2],
                'weight': float(row[3]),
                'unit': row[4]
            })

# Sort by timestamp
measurements.sort(key=lambda x: x['timestamp'])

print(f"User {user_id} weight history:")
print("=" * 80)
print(f"Total measurements: {len(measurements)}")
print()

# Show all measurements
print("Date                 Source                          Weight (kg)")
print("-" * 80)
for m in measurements:
    print(f"{m['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  {m['source']:30s}  {m['weight']:8.2f}")

# Analyze the pattern
weights = [m['weight'] for m in measurements]
print("\n" + "=" * 80)
print("Statistical Analysis:")
print(f"  Mean: {np.mean(weights):.2f} kg")
print(f"  Median: {np.median(weights):.2f} kg")
print(f"  Std Dev: {np.std(weights):.2f} kg")
print(f"  Min: {np.min(weights):.2f} kg")
print(f"  Max: {np.max(weights):.2f} kg")
print(f"  Range: {np.max(weights) - np.min(weights):.2f} kg")

# Look for outliers
q1 = np.percentile(weights, 25)
q3 = np.percentile(weights, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"\nOutlier Detection (IQR method):")
print(f"  Q1: {q1:.2f} kg")
print(f"  Q3: {q3:.2f} kg")
print(f"  IQR: {iqr:.2f} kg")
print(f"  Lower bound: {lower_bound:.2f} kg")
print(f"  Upper bound: {upper_bound:.2f} kg")

outliers = [m for m in measurements if m['weight'] < lower_bound or m['weight'] > upper_bound]
print(f"\nOutliers found: {len(outliers)}")
for m in outliers:
    print(f"  {m['timestamp'].strftime('%Y-%m-%d')} - {m['weight']:.2f} kg ({m['source']})")

# Check for unit conversion issues
print("\n" + "=" * 80)
print("Checking for potential unit conversion issues:")
for m in measurements:
    # Check if weight might be in pounds mistakenly entered as kg
    potential_kg = m['weight'] / 2.20462
    if 70 <= potential_kg <= 110:  # Reasonable weight range
        print(f"  {m['timestamp'].strftime('%Y-%m-%d')} - {m['weight']:.2f} might be lbs? ({potential_kg:.2f} kg)")
