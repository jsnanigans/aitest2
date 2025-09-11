#!/usr/bin/env python3
"""
Fix for unit conversion issues - demonstrates the problem and solution
"""

import sys
sys.path.insert(0, '.')

import csv
from datetime import datetime
import numpy as np
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
import tomllib

def convert_weight_to_kg(weight_str, unit_str):
    """
    Convert weight to kg based on unit.
    Returns None if not a valid weight measurement.
    """
    weight = float(weight_str)
    unit = (unit_str or 'kg').lower().strip()
    
    # Handle pounds
    if 'lb' in unit:
        return weight * 0.453592  # Convert lbs to kg
    
    # Handle BSA (body surface area) - not a weight!
    if 'm2' in unit or 'm²' in unit:
        return None  # This is BSA, not weight
    
    # Handle kg (default)
    if 'kg' in unit or unit == '':
        return weight
    
    # Unknown unit - safer to skip
    print(f"WARNING: Unknown unit '{unit}' for weight {weight}")
    return None

def process_user_with_unit_conversion(user_id, csv_path, config):
    """Process user with proper unit conversion"""
    
    measurements = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == user_id:
                weight_str = row.get("weight", "").strip()
                unit_str = row.get("unit", "kg")
                
                if weight_str and weight_str.upper() != "NULL":
                    try:
                        weight_kg = convert_weight_to_kg(weight_str, unit_str)
                        if weight_kg is not None:
                            timestamp = datetime.strptime(row.get('effectiveDateTime'), "%Y-%m-%d %H:%M:%S")
                            measurements.append((timestamp, weight_kg, weight_str, unit_str))
                    except Exception as e:
                        continue
    
    measurements.sort(key=lambda x: x[0])
    return measurements

def main():
    user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Get measurements with proper conversion
    measurements = process_user_with_unit_conversion(user_id, csv_path, config)
    
    print(f"Found {len(measurements)} valid WEIGHT measurements (after filtering BSA and converting units)")
    
    # Show converted weights
    print("\nFirst 30 converted measurements:")
    for i, (timestamp, weight_kg, orig_weight, orig_unit) in enumerate(measurements[:30]):
        conversion_note = ""
        if 'lb' in (orig_unit or '').lower():
            conversion_note = f" (converted from {orig_weight} {orig_unit})"
        print(f"  {i+1:3d}. {timestamp.date()} : {weight_kg:7.2f} kg{conversion_note}")
    
    # Analyze cleaned data
    weights = [w for _, w, _, _ in measurements]
    print(f"\n=== CLEANED Data Statistics ===")
    print(f"  Mean: {np.mean(weights):.2f} kg")
    print(f"  Std Dev: {np.std(weights):.2f} kg")
    print(f"  Min: {np.min(weights):.2f} kg")
    print(f"  Max: {np.max(weights):.2f} kg")
    
    # Process through Kalman with cleaned data
    db = ProcessorStateDB()
    results = []
    
    print(f"\nProcessing through Kalman filter with CLEANED data:")
    for i, (timestamp, weight, orig_weight, orig_unit) in enumerate(measurements[:30]):
        result = WeightProcessor.process_weight(
            user_id=user_id + "_fixed",  # Use different ID to avoid cached state
            weight=weight,
            timestamp=timestamp,
            source="csv",
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result and 'filtered_weight' in result:
            results.append(result)
            innovation = result['raw_weight'] - result['filtered_weight']
            if abs(innovation) > 2.0:
                print(f"  {i+1}. {timestamp.date()}: raw={weight:.1f}, filtered={result['filtered_weight']:.1f}, diff={innovation:.1f}")
    
    # Compare volatility
    if results:
        innovations = [abs(r['raw_weight'] - r['filtered_weight']) for r in results]
        print(f"\n=== Kalman Filter Performance ===")
        print(f"  Average absolute innovation: {np.mean(innovations):.2f} kg")
        print(f"  Max absolute innovation: {np.max(innovations):.2f} kg")
        print(f"  Measurements with >2kg innovation: {sum(1 for i in innovations if i > 2.0)}")

if __name__ == "__main__":
    main()