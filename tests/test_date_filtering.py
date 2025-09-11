#!/usr/bin/env python3
"""Test date filtering functionality in main.py"""

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from main import stream_process


def test_date_filtering():
    """Test that date filtering correctly filters out measurements outside the specified range."""
    
    # Create test data with dates spanning before 2020 to after 2025-09-05
    test_data = [
        ["user_id", "weight", "effectiveDateTime", "source_type", "unit"],
        # Before 2020 - should be filtered
        ["user1", "70.0", "2019-12-31", "scale", "kg"],
        ["user1", "70.5", "2019-06-15", "scale", "kg"],
        # In valid range - should be processed
        ["user1", "71.0", "2020-01-01", "scale", "kg"],
        ["user1", "71.5", "2021-06-15", "scale", "kg"],
        ["user1", "72.0", "2023-03-20", "scale", "kg"],
        ["user1", "72.5", "2025-09-05", "scale", "kg"],
        # After 2025-09-05 - should be filtered
        ["user1", "73.0", "2025-09-06", "scale", "kg"],
        ["user1", "73.5", "2025-12-31", "scale", "kg"],
        ["user1", "74.0", "2026-01-01", "scale", "kg"],
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(test_data)
        csv_path = f.name
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        # Test configuration with date filters
        config = {
            "data": {
                "csv_file": csv_path,
                "output_dir": output_dir,
                "max_users": 0,
                "user_offset": 0,
                "min_readings": 0,
                "test_users": [],
                "min_date": "2020-01-01",
                "max_date": "2025-09-05"
            },
            "processing": {
                "min_weight": 30.0,
                "max_weight": 400.0,
                "max_daily_change": 0.05,
                "extreme_threshold": 0.20,
                "kalman_cleanup_threshold": 4.0
            },
            "kalman": {
                "initial_variance": 0.5,
                "transition_covariance_weight": 0.05,
                "transition_covariance_trend": 0.0005,
                "observation_covariance": 1.5,
                "reset_gap_days": 30
            },
            "visualization": {
                "enabled": False
            },
            "logging": {
                "progress_interval": 10000,
                "timestamp_format": "test_%Y%m%d_%H%M%S"
            },
            "use_enhanced": False
        }
        
        # Run processing
        results, stats = stream_process(csv_path, output_dir, config)
        
        # Check results
        print(f"\nTest Results:")
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  Date filtered: {stats.get('date_filtered', 0)}")
        print(f"  Accepted: {stats['accepted']}")
        print(f"  Rejected: {stats['rejected']}")
        
        # Verify filtering worked correctly
        assert stats['total_rows'] == 9, f"Expected 9 total rows, got {stats['total_rows']}"
        # 2 before 2020-01-01 + 3 after 2025-09-05 = 5 filtered
        assert stats.get('date_filtered', 0) == 5, f"Expected 5 filtered rows, got {stats.get('date_filtered', 0)}"
        
        # Should have processed 4 valid measurements (2020-01-01 to 2025-09-05 inclusive)
        total_processed = stats['accepted'] + stats['rejected']
        assert total_processed == 4, f"Expected 4 processed measurements, got {total_processed}"
        
        print("\n✅ Date filtering test passed!")
        
        # Test with no date filters
        config["data"]["min_date"] = ""
        config["data"]["max_date"] = ""
        
        results2, stats2 = stream_process(csv_path, output_dir, config)
        
        print(f"\nTest without filters:")
        print(f"  Total rows: {stats2['total_rows']}")
        print(f"  Date filtered: {stats2.get('date_filtered', 0)}")
        print(f"  Accepted: {stats2['accepted']}")
        print(f"  Rejected: {stats2['rejected']}")
        
        # Should process all measurements when no filters
        assert stats2.get('date_filtered', 0) == 0, f"Expected 0 filtered rows without filters, got {stats2.get('date_filtered', 0)}"
        total_processed2 = stats2['accepted'] + stats2['rejected']
        # We have 9 data rows total (excluding header)
        assert total_processed2 == 9, f"Expected 9 processed measurements without filters, got {total_processed2}"
        
        print("\n✅ No filter test passed!")
    
    # Clean up temp file
    Path(csv_path).unlink()
    
    print("\n✅ All date filtering tests passed!")


if __name__ == "__main__":
    test_date_filtering()