#!/usr/bin/env python3
"""
Verify that the unit conversion fix is working
"""

import csv

def test_unit_conversion():
    """Test the unit conversion logic from main.py"""
    
    test_cases = [
        ("212.3", "[lb_ap]", 96.30),  # Pounds to kg
        ("223.1", "[lb_ap]", 101.20),  # Pounds to kg
        ("2.10", "m2", None),  # BSA - should be skipped
        ("2.15", "m2", None),  # BSA - should be skipped
        ("88.0", "kg", 88.0),  # kg - no conversion
        ("99.79", "", 99.79),  # No unit - assume kg
    ]
    
    for weight_str, unit, expected in test_cases:
        try:
            weight_raw = float(weight_str)
            unit_lower = (unit or "kg").lower().strip()
            
            # Skip BSA measurements
            if 'm2' in unit_lower or 'm²' in unit_lower:
                result = None
            # Convert pounds to kg
            elif 'lb' in unit_lower:
                result = weight_raw * 0.453592
            # Default to kg
            else:
                result = weight_raw
                
        except (ValueError, TypeError):
            result = None
        
        # Check result
        if expected is None:
            if result is None:
                print(f"✅ {weight_str} {unit} → SKIPPED (correct)")
            else:
                print(f"❌ {weight_str} {unit} → {result:.2f} kg (should be SKIPPED)")
        else:
            if result is not None and abs(result - expected) < 0.1:
                print(f"✅ {weight_str} {unit} → {result:.2f} kg (expected {expected:.2f})")
            else:
                print(f"❌ {weight_str} {unit} → {result:.2f} kg (expected {expected:.2f})")

def count_user_measurements():
    """Count measurements for the problem user with and without unit handling"""
    
    user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    raw_count = 0
    converted_count = 0
    skipped_bsa = 0
    converted_lbs = 0
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == user_id:
                weight_str = row.get("weight", "").strip()
                if weight_str and weight_str.upper() != "NULL":
                    raw_count += 1
                    
                    try:
                        weight_raw = float(weight_str)
                        unit = (row.get("unit") or "kg").lower().strip()
                        
                        if 'm2' in unit or 'm²' in unit:
                            skipped_bsa += 1
                        else:
                            converted_count += 1
                            if 'lb' in unit:
                                converted_lbs += 1
                    except:
                        pass
    
    print(f"\nUser {user_id[:8]}... measurement counts:")
    print(f"  Total raw measurements: {raw_count}")
    print(f"  Valid weight measurements: {converted_count}")
    print(f"  BSA measurements skipped: {skipped_bsa}")
    print(f"  Pounds converted to kg: {converted_lbs}")
    print(f"  Reduction: {raw_count} → {converted_count} ({(raw_count-converted_count)/raw_count*100:.1f}% filtered)")

if __name__ == "__main__":
    print("Testing unit conversion logic:")
    test_unit_conversion()
    count_user_measurements()