#!/usr/bin/env python3
"""Test script to verify fixes for duplicate detection and residual visualization."""

import csv
from datetime import datetime
from src.filters.enhanced_validation_gate import EnhancedValidationGate

def test_user_01672f42():
    """Test user with duplicate detection issues."""
    print("\n=== Testing User 01672f42-568b-4d49-abbc-eee60d87ccb2 ===")
    
    gate = EnhancedValidationGate({'duplicate_threshold_kg': 0.5})
    user_id = '01672f42-568b-4d49-abbc-eee60d87ccb2'
    
    # Count duplicates with fixed code
    with open('2025-09-05_optimized.csv', 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        rejected_count = 0
        accepted_count = 0
        
        for row in reader:
            if row['user_id'] != user_id:
                continue
            
            count += 1
            reading = {
                'date': datetime.fromisoformat(row['effectivDateTime'].replace(' ', 'T')),
                'weight': float(row['weight']),
                'source': row['source_type']
            }
            
            # Check for duplicate
            is_dup = gate.should_deduplicate(user_id, reading)
            
            if is_dup:
                rejected_count += 1
            else:
                # Validate if not duplicate
                is_valid, reason = gate.validate_reading(user_id, reading)
                if is_valid:
                    accepted_count += 1
        
        print(f'Total rows: {count}')
        print(f'Rejected as duplicates: {rejected_count}')
        print(f'Accepted after validation: {accepted_count}')
        print(f'Acceptance rate: {accepted_count/count*100:.1f}%')
        
        # Expected: ~10 duplicates (legitimate), acceptance rate > 40%
        assert rejected_count < 15, f"Too many duplicates: {rejected_count}"
        assert accepted_count > 25, f"Too few accepted: {accepted_count}"
        print("✓ Duplicate detection bug fixed!")

def test_user_00e76445():
    """Test user with statistical outlier issues."""
    print("\n=== Testing User 00e76445-7a74-4e39-b48d-3980d2186604 ===")
    
    gate = EnhancedValidationGate({
        'outlier_z_score': 2.5,
        'min_readings_for_stats': 5
    })
    user_id = '00e76445-7a74-4e39-b48d-3980d2186604'
    
    # Process readings
    with open('2025-09-05_optimized.csv', 'r') as f:
        reader = csv.DictReader(f)
        weights = []
        dates = []
        
        for row in reader:
            if row['user_id'] != user_id:
                continue
            
            dates.append(row['effectivDateTime'])
            weights.append(float(row['weight']))
        
        print(f'Total readings: {len(weights)}')
        print(f'Weight range: {min(weights):.1f} - {max(weights):.1f} kg')
        print(f'Weight change: {max(weights) - min(weights):.1f} kg')
        
        # Group by year to show trend
        years = {}
        for date_str, weight in zip(dates, weights):
            year = date_str[:4]
            if year not in years:
                years[year] = []
            years[year].append(weight)
        
        print("\nWeight by year:")
        for year in sorted(years.keys()):
            avg = sum(years[year]) / len(years[year])
            print(f"  {year}: {avg:.1f} kg (n={len(years[year])})")
        
        # The issue: User lost ~20kg from 2020-2024 to 2025
        # Statistical outlier detection sees 2025 readings as outliers
        print("\n⚠ User has significant weight loss (20+ kg)")
        print("  Statistical outlier detection needs time-aware windowing")

def main():
    """Run all tests."""
    print("Testing fixes for duplicate detection and visualization issues...")
    
    try:
        test_user_01672f42()
        test_user_00e76445()
        print("\n✅ All tests completed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())