#!/usr/bin/env python3

import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.preprocess_csv import preprocess_csv, parse_datetime


def test_basic_preprocessing():
    print("Testing basic preprocessing...")
    
    config = {
        'ignore_sources': [],
        'min_datetime': '',
        'max_datetime': '',
        'max_buffer_size': 10000,
        'output_suffix': '_test',
        'verbose': False,
        'skip_duplicates': True,
    }
    
    input_file = Path(__file__).parent.parent / 'data' / 'test_sample.csv'
    output_file = Path(__file__).parent.parent / 'data' / 'test_sample_test.csv'
    
    stats = preprocess_csv(str(input_file), str(output_file), config)
    
    assert stats['total_rows'] == 99, f"Expected 99 rows, got {stats['total_rows']}"
    assert stats['users_processed'] == 5, f"Expected 5 users, got {stats['users_processed']}"
    assert stats['output_rows'] == 99, f"Expected 99 output rows, got {stats['output_rows']}"
    
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        current_user = None
        last_date = None
        
        for row in rows:
            user_id = row['user_id']
            date_str = row['effectiveDateTime']
            date = parse_datetime(date_str)
            
            if user_id != current_user:
                current_user = user_id
                last_date = date
            else:
                assert date >= last_date, f"Dates not sorted for user {user_id}: {date} < {last_date}"
                last_date = date
    
    output_file.unlink()
    print("✓ Basic preprocessing test passed")


def test_date_filtering():
    print("Testing date filtering...")
    
    config = {
        'ignore_sources': [],
        'min_datetime': '2022-01-01',
        'max_datetime': '2023-12-31',
        'max_buffer_size': 10000,
        'output_suffix': '_datetest',
        'verbose': False,
        'skip_duplicates': True,
    }
    
    input_file = Path(__file__).parent.parent / 'data' / 'test_sample.csv'
    output_file = Path(__file__).parent.parent / 'data' / 'test_sample_datetest.csv'
    
    stats = preprocess_csv(str(input_file), str(output_file), config)
    
    assert stats['filtered_by_date'] > 0, "Should have filtered some dates"
    
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = parse_datetime(row['effectiveDateTime'])
            assert date >= datetime(2022, 1, 1), f"Date {date} before min"
            assert date <= datetime(2023, 12, 31, 23, 59, 59), f"Date {date} after max"
    
    output_file.unlink()
    print("✓ Date filtering test passed")


def test_source_filtering():
    print("Testing source filtering...")
    
    config = {
        'ignore_sources': ['internal-questionnaire', 'patient-upload'],
        'min_datetime': '',
        'max_datetime': '',
        'max_buffer_size': 10000,
        'output_suffix': '_sourcetest',
        'verbose': False,
        'skip_duplicates': True,
    }
    
    input_file = Path(__file__).parent.parent / 'data' / 'test_sample.csv'
    output_file = Path(__file__).parent.parent / 'data' / 'test_sample_sourcetest.csv'
    
    stats = preprocess_csv(str(input_file), str(output_file), config)
    
    assert stats['filtered_by_source'] > 0, "Should have filtered some sources"
    
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row['source_type'].lower()
            assert 'internal-questionnaire' not in source
            assert 'patient-upload' not in source
    
    output_file.unlink()
    print("✓ Source filtering test passed")


def test_duplicate_removal():
    print("Testing duplicate removal...")
    
    test_csv = Path(__file__).parent.parent / 'data' / 'test_duplicates.csv'
    with open(test_csv, 'w') as f:
        f.write("user_id,effectiveDateTime,source_type,weight,unit\n")
        f.write("user1,2024-01-01 10:00:00,device,75.5,kg\n")
        f.write("user1,2024-01-01 10:00:00,device,75.5,kg\n")
        f.write("user1,2024-01-02 10:00:00,device,75.6,kg\n")
        f.write("user1,2024-01-02 10:00:00,device,75.6,kg\n")
        f.write("user1,2024-01-02 10:00:00,device,75.7,kg\n")
    
    config = {
        'ignore_sources': [],
        'min_datetime': '',
        'max_datetime': '',
        'max_buffer_size': 10000,
        'output_suffix': '_duptest',
        'verbose': False,
        'skip_duplicates': True,
    }
    
    output_file = Path(__file__).parent.parent / 'data' / 'test_duplicates_duptest.csv'
    
    stats = preprocess_csv(str(test_csv), str(output_file), config)
    
    assert stats['duplicates_removed'] == 2, f"Expected 2 duplicates removed, got {stats['duplicates_removed']}"
    assert stats['output_rows'] == 3, f"Expected 3 output rows, got {stats['output_rows']}"
    
    test_csv.unlink()
    output_file.unlink()
    print("✓ Duplicate removal test passed")


if __name__ == '__main__':
    print("\nRunning preprocessor tests...\n")
    
    try:
        test_basic_preprocessing()
        test_date_filtering()
        test_source_filtering()
        test_duplicate_removal()
        
        print("\n✅ All tests passed!\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)