#!/usr/bin/env python3
"""
Test unit conversion for user with mixed BSA/pound/kg measurements
"""

import csv
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_user_data(csv_file: str, user_id: str):
    """Analyze raw data for a specific user to understand unit issues."""
    
    measurements = []
    
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") != user_id:
                continue
            
            weight_str = row.get("weight", "").strip()
            if not weight_str or weight_str.upper() == "NULL":
                continue
            
            try:
                weight_raw = float(weight_str)
            except (ValueError, TypeError):
                continue
            
            unit = (row.get("unit") or "").lower().strip()
            source = row.get("source_type") or row.get("source", "unknown")
            date_str = row.get("effectiveDateTime")
            
            measurements.append({
                'timestamp': date_str,
                'weight_raw': weight_raw,
                'unit': unit,
                'source': source
            })
    
    print(f"\nUser {user_id} - Raw Data Analysis")
    print("=" * 80)
    print(f"Total measurements: {len(measurements)}")
    
    # Group by source
    by_source = defaultdict(list)
    for m in measurements:
        by_source[m['source']].append(m)
    
    print("\nMeasurements by source:")
    for source, items in sorted(by_source.items()):
        weights = [item['weight_raw'] for item in items]
        units = set(item['unit'] for item in items)
        print(f"  {source:20s}: {len(items):3d} measurements, "
              f"range {min(weights):6.1f} - {max(weights):6.1f}, "
              f"units: {units or {''}}")
    
    # Identify problematic measurements
    print("\nProblematic measurements:")
    bsa_count = 0
    pound_count = 0
    
    for m in measurements:
        # BSA detection
        if 'BSA' in m['source'].upper() or 'm2' in m['unit'] or 'm²' in m['unit']:
            bsa_count += 1
            if bsa_count <= 3:  # Show first few
                print(f"  BSA: {m['timestamp'][:10]} - {m['weight_raw']:.2f} {m['unit']} (source: {m['source']})")
        
        # Pound detection
        elif 'lb' in m['unit'] or 'pound' in m['unit']:
            pound_count += 1
            if pound_count <= 3:
                print(f"  LBS: {m['timestamp'][:10]} - {m['weight_raw']:.1f} {m['unit']} → {m['weight_raw']*0.453592:.1f} kg")
        
        # Heuristic pound detection (no unit but value > 130)
        elif m['weight_raw'] > 130 and m['weight_raw'] < 400 and not m['unit']:
            pound_count += 1
            if pound_count <= 3:
                print(f"  LBS?: {m['timestamp'][:10]} - {m['weight_raw']:.1f} (no unit) → {m['weight_raw']*0.453592:.1f} kg")
    
    if bsa_count > 3:
        print(f"  ... and {bsa_count - 3} more BSA measurements")
    if pound_count > 3:
        print(f"  ... and {pound_count - 3} more pound measurements")
    
    print(f"\nSummary:")
    print(f"  BSA measurements to skip: {bsa_count}")
    print(f"  Pound measurements to convert: {pound_count}")
    print(f"  Valid kg measurements: {len(measurements) - bsa_count - pound_count}")
    
    # Calculate expected clean weight after conversion
    valid_weights = []
    for m in measurements:
        if 'BSA' in m['source'].upper() or 'm2' in m['unit'] or 'm²' in m['unit']:
            continue
        elif 'lb' in m['unit'] or (m['weight_raw'] > 130 and m['weight_raw'] < 400 and not m['unit']):
            valid_weights.append(m['weight_raw'] * 0.453592)
        else:
            valid_weights.append(m['weight_raw'])
    
    if valid_weights:
        print(f"\nExpected weight after proper conversion:")
        print(f"  Mean: {sum(valid_weights)/len(valid_weights):.1f} kg")
        print(f"  Range: {min(valid_weights):.1f} - {max(valid_weights):.1f} kg")
        print(f"  Valid measurements: {len(valid_weights)} of {len(measurements)}")


if __name__ == "__main__":
    # Test with the problematic user
    csv_file = "data/test_sample.csv"
    test_user = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        test_user = sys.argv[2]
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)
    
    analyze_user_data(csv_file, test_user)