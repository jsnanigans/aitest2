#!/usr/bin/env python3
"""
Investigate raw data for user 1e87d3ab-20b1-479d-ad4d-8986e1af38da
"""

import sys
sys.path.insert(0, '.')

import csv
from datetime import datetime
import numpy as np

def main():
    user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    measurements = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == user_id:
                weight_str = row.get("weight", "").strip()
                if weight_str and weight_str.upper() != "NULL":
                    try:
                        weight = float(weight_str)
                        timestamp = datetime.strptime(row.get('effectiveDateTime'), "%Y-%m-%d %H:%M:%S")
                        unit = row.get('unit', 'kg')
                        measurements.append((timestamp, weight, unit))
                    except Exception as e:
                        print(f"Error parsing: {row} - {e}")
                        continue
    
    measurements.sort(key=lambda x: x[0])
    print(f"Found {len(measurements)} measurements for user {user_id}")
    
    # Show all measurements to spot outliers
    print("\nAll measurements:")
    for i, (timestamp, weight, unit) in enumerate(measurements):
        flag = ""
        if weight < 30 or weight > 200:
            flag = " *** OUTLIER ***"
        print(f"  {i+1:3d}. {timestamp.date()} : {weight:7.2f} {unit}{flag}")
    
    # Analyze without extreme outliers
    weights = [w for _, w, _ in measurements]
    normal_weights = [w for w in weights if 30 <= w <= 200]
    
    print(f"\n=== Raw Data Analysis ===")
    print(f"ALL weights ({len(weights)} total):")
    print(f"  Mean: {np.mean(weights):.2f} kg")
    print(f"  Std Dev: {np.std(weights):.2f} kg")
    print(f"  Min: {np.min(weights):.2f} kg")
    print(f"  Max: {np.max(weights):.2f} kg")
    
    if normal_weights:
        print(f"\nNORMAL range weights (30-200 kg, {len(normal_weights)} total):")
        print(f"  Mean: {np.mean(normal_weights):.2f} kg")
        print(f"  Std Dev: {np.std(normal_weights):.2f} kg")
        print(f"  Min: {np.min(normal_weights):.2f} kg")
        print(f"  Max: {np.max(normal_weights):.2f} kg")
    
    # Find extreme outliers
    outliers = [(i, t, w, u) for i, (t, w, u) in enumerate(measurements) if w < 30 or w > 200]
    if outliers:
        print(f"\n=== EXTREME OUTLIERS ({len(outliers)} found) ===")
        for i, timestamp, weight, unit in outliers:
            print(f"  Position {i+1}: {timestamp.date()} - {weight:.2f} {unit}")
    
    # Look at recent data (last 3 months)
    if measurements:
        latest = measurements[-1][0]
        three_months_ago = datetime(latest.year, latest.month - 3 if latest.month > 3 else latest.month + 9, latest.day)
        if latest.month <= 3:
            three_months_ago = three_months_ago.replace(year=latest.year - 1)
        
        recent = [(t, w, u) for t, w, u in measurements if t >= three_months_ago]
        if recent:
            recent_weights = [w for _, w, _ in recent]
            print(f"\n=== RECENT DATA (last 3 months, {len(recent)} measurements) ===")
            print(f"  Mean: {np.mean(recent_weights):.2f} kg")
            print(f"  Std Dev: {np.std(recent_weights):.2f} kg")
            print(f"  Min: {np.min(recent_weights):.2f} kg")
            print(f"  Max: {np.max(recent_weights):.2f} kg")

if __name__ == "__main__":
    main()