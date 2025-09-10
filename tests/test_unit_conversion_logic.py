#!/usr/bin/env python3
"""
Test unit conversion logic in main.py
"""

import tempfile
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import stream_process, load_config

def create_test_csv_with_mixed_units():
    """Create a test CSV with mixed units to verify conversion."""
    
    test_data = [
        # Headers
        ['user_id', 'effectiveDateTime', 'source_type', 'weight', 'unit'],
        
        # User with mixed units
        ['test-user-1', '2024-01-01 10:00:00', 'scale', '88.5', 'kg'],  # Good kg
        ['test-user-1', '2024-01-02 10:00:00', 'BSA', '1.95', 'm2'],  # BSA - should skip
        ['test-user-1', '2024-01-03 10:00:00', 'scale', '195', 'lbs'],  # Pounds - should convert to ~88.5 kg
        ['test-user-1', '2024-01-04 10:00:00', 'scale', '88.3', 'kg'],  # Good kg
        ['test-user-1', '2024-01-05 10:00:00', 'BSA', '1.94', 'm²'],  # BSA - should skip
        ['test-user-1', '2024-01-06 10:00:00', 'scale', '194.5', 'pounds'],  # Pounds - should convert
        ['test-user-1', '2024-01-07 10:00:00', 'scale', '88.2', 'kg'],  # Good kg
        ['test-user-1', '2024-01-08 10:00:00', 'manual', '195.5', ''],  # No unit but > 130 - likely pounds
        ['test-user-1', '2024-01-09 10:00:00', 'scale', '88.4', 'kg'],  # Good kg
        ['test-user-1', '2024-01-10 10:00:00', 'BSA-calc', '1.96', ''],  # BSA by source name
        ['test-user-1', '2024-01-11 10:00:00', 'scale', '88.1', 'kg'],  # Good kg
        ['test-user-1', '2024-01-12 10:00:00', 'scale', '88.3', 'kg'],  # Good kg
    ]
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    writer = csv.writer(temp_file)
    writer.writerows(test_data)
    temp_file.close()
    
    return temp_file.name

def test_unit_conversion():
    """Test that unit conversion works correctly."""
    
    print("Testing unit conversion logic...")
    print("=" * 60)
    
    # Create test CSV
    csv_file = create_test_csv_with_mixed_units()
    print(f"Created test CSV: {csv_file}")
    
    # Load config and override for testing
    config = load_config("config.toml")
    config["data"]["test_users"] = ["test-user-1"]
    config["visualization"]["enabled"] = False
    
    # Create temp output dir
    output_dir = tempfile.mkdtemp()
    
    # Process the data
    print("\nProcessing test data...")
    results, stats = stream_process(csv_file, output_dir, config)
    
    print("\nResults Analysis:")
    print("-" * 40)
    
    # Check results for our test user
    if "test-user-1" in results:
        user_results = results["test-user-1"]
        print(f"Total results for test-user-1: {len(user_results)}")
        
        # Analyze accepted weights
        accepted_weights = [r['filtered_weight'] for r in user_results if r['accepted'] and r.get('filtered_weight')]
        raw_weights = [r['raw_weight'] for r in user_results if r['accepted']]
        
        print(f"Accepted measurements: {len(accepted_weights)}")
        print(f"Raw accepted weights: {[f'{w:.1f}' for w in raw_weights]}")
        
        if accepted_weights:
            print(f"Filtered weight range: {min(accepted_weights):.1f} - {max(accepted_weights):.1f} kg")
            print(f"Average filtered weight: {sum(accepted_weights)/len(accepted_weights):.1f} kg")
        
        # Check that BSA measurements were skipped
        sources = [r['source'] for r in user_results]
        bsa_count = sum(1 for s in sources if 'BSA' in s.upper())
        print(f"\nBSA measurements in results: {bsa_count} (should be 0)")
        
        # Check weight consistency
        if accepted_weights:
            weight_variance = max(accepted_weights) - min(accepted_weights)
            print(f"Weight variance: {weight_variance:.2f} kg")
            
            if weight_variance < 2.0:  # Should be very consistent after conversion
                print("✓ Weights are consistent after unit conversion")
            else:
                print("✗ Weights show high variance - conversion may have failed")
        
        # Show any rejected measurements
        rejected = [r for r in user_results if not r['accepted']]
        if rejected:
            print(f"\nRejected measurements: {len(rejected)}")
            for r in rejected[:3]:
                print(f"  - {r.get('rejection_reason', 'Unknown reason')}")
    else:
        print("ERROR: Test user not found in results!")
    
    # Clean up
    Path(csv_file).unlink()
    print(f"\nTest complete. Stats: {stats}")

if __name__ == "__main__":
    test_unit_conversion()