#!/usr/bin/env python3
"""Debug script to check source_type handling in the pipeline."""

import csv
from pathlib import Path
from datetime import datetime
from src.processing.user_processor import UserProcessor
from src.core.types import WeightMeasurement

def test_source_extraction():
    """Test how source_type is extracted from CSV rows."""
    
    # Test cases with different source configurations
    test_rows = [
        {'source_type': 'patient-device', 'weight': '70', 'effectiveDateTime': '2025-01-01', 'user_id': 'test1'},
        {'source_type': '', 'weight': '71', 'effectiveDateTime': '2025-01-02', 'user_id': 'test1'},  # Empty source
        {'weight': '72', 'effectiveDateTime': '2025-01-03', 'user_id': 'test1'},  # Missing source_type
        {'source': 'care-team-upload', 'weight': '73', 'effectiveDateTime': '2025-01-04', 'user_id': 'test1'},  # Only 'source' field
        {'source_type': None, 'weight': '74', 'effectiveDateTime': '2025-01-05', 'user_id': 'test1'},  # None value
    ]
    
    print("Testing source_type extraction:")
    print("-" * 60)
    
    for i, row in enumerate(test_rows, 1):
        # Extract source the same way UserProcessor does
        source_value = row.get('source_type') or row.get('source')
        
        print(f"Test {i}:")
        print(f"  Row: {row}")
        print(f"  Extracted source: '{source_value}' (type: {type(source_value).__name__})")
        print(f"  Will display as: {'Other' if not source_value or source_value == 'unknown' else source_value}")
        print()

def check_csv_for_empty_sources():
    """Check actual CSV file for empty source_type values."""
    csv_files = list(Path('.').glob('*_optimized.csv'))
    
    if not csv_files:
        print("No optimized CSV files found")
        return
    
    csv_file = csv_files[0]
    print(f"\nChecking CSV file: {csv_file}")
    print("-" * 60)
    
    empty_count = 0
    total_count = 0
    source_counts = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            source = row.get('source_type') or row.get('source') or 'unknown'
            
            if not row.get('source_type'):
                empty_count += 1
                if empty_count <= 5:  # Show first 5 examples
                    print(f"  Row {total_count}: Empty source_type, user_id={row.get('user_id')[:8]}...")
            
            source_counts[source] = source_counts.get(source, 0) + 1
            
            if total_count >= 10000:  # Sample first 10k rows
                break
    
    print(f"\nSource distribution (first {total_count} rows):")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_count) * 100
        print(f"  {source:30s}: {count:6d} ({pct:5.1f}%)")
    
    if empty_count > 0:
        print(f"\nFound {empty_count} rows with empty/missing source_type ({empty_count/total_count*100:.1f}%)")

def test_processor_pipeline():
    """Test the full processor pipeline with different source values."""
    from src.core.config_loader import load_config
    
    config = load_config()
    processor = UserProcessor('test_user', config)
    
    test_data = [
        {'user_id': 'test_user', 'weight': '70', 'effectiveDateTime': '2025-01-01', 'source_type': 'patient-device'},
        {'user_id': 'test_user', 'weight': '71', 'effectiveDateTime': '2025-01-02', 'source_type': ''},  # Empty
        {'user_id': 'test_user', 'weight': '72', 'effectiveDateTime': '2025-01-03'},  # Missing
    ]
    
    print("\nTesting processor pipeline:")
    print("-" * 60)
    
    for row in test_data:
        processor.process_row(row)
    
    result = processor.get_results()
    
    if result and 'time_series' in result:
        for ts in result['time_series']:
            print(f"Date: {ts['date']}, Source: '{ts.get('source')}', Weight: {ts['weight']:.1f}kg")
    
    print(f"\nTotal processed: {len(processor.raw_measurements)} measurements")

if __name__ == '__main__':
    print("=" * 60)
    print("SOURCE TYPE DEBUG SCRIPT")
    print("=" * 60)
    
    test_source_extraction()
    check_csv_for_empty_sources()
    test_processor_pipeline()