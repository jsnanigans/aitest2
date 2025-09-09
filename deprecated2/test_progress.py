#!/usr/bin/env python3
"""
Test the improved CLI progress indication.
"""

import time
import csv
from pathlib import Path
from datetime import datetime, timedelta
import random

from src.core.progress import ProgressIndicator, ProcessingStatus


def test_progress_bar():
    """Test progress bar with known total."""
    ProcessingStatus.section("Testing Progress Bar")
    
    total_items = 100
    progress = ProgressIndicator(total_items=total_items, show_eta=True)
    progress.start("Processing items")
    
    for i in range(1, total_items + 1):
        # Simulate work
        time.sleep(0.02)
        
        # Update progress with statistics
        progress.update(i, {
            'rows_processed': i * 10,
            'users_processed': i // 5,
            'current_user': f"user_{i:03d}",
            'accepted': i * 8,
            'rejected': i * 2
        })
    
    progress.finish("All items processed")


def test_spinner():
    """Test spinner for unknown total."""
    ProcessingStatus.section("Testing Spinner")
    
    progress = ProgressIndicator(total_items=None)  # No total = spinner mode
    progress.start("Processing stream")
    
    for i in range(50):
        # Simulate work
        time.sleep(0.05)
        
        # Update with statistics
        progress.update(i, {
            'rows_processed': i * 100,
            'users_processed': i * 2,
            'current_user': f"user_{random.randint(1000, 9999)}"
        })
    
    progress.finish("Stream processing complete")


def test_status_messages():
    """Test colored status messages."""
    ProcessingStatus.section("Testing Status Messages")
    
    ProcessingStatus.info("This is an informational message")
    ProcessingStatus.success("This operation succeeded")
    ProcessingStatus.warning("This is a warning about something")
    ProcessingStatus.error("This is an error message")
    ProcessingStatus.processing("Currently processing data...")
    
    print("\nSimulating different acceptance rates:")
    
    # High acceptance rate (green)
    ProcessingStatus.success("Acceptance rate: 95.2%")
    
    # Medium acceptance rate (yellow)
    ProcessingStatus.warning("Acceptance rate: 75.8%")
    
    # Low acceptance rate (red)
    ProcessingStatus.error("Low acceptance rate: 45.3%")


def test_real_processing():
    """Simulate real CSV processing with progress."""
    ProcessingStatus.section("Simulating Real Processing")
    
    # Create test CSV
    test_file = Path("test_progress_data.csv")
    ProcessingStatus.processing(f"Creating test file: {test_file}")
    
    with open(test_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'weight', 'date', 'source'])
        writer.writeheader()
        
        # Write test data
        for user_num in range(5):
            user_id = f"user_{user_num:03d}"
            for day in range(20):
                writer.writerow({
                    'user_id': user_id,
                    'weight': 70 + random.uniform(-2, 2),
                    'date': (datetime.now() - timedelta(days=day)).isoformat(),
                    'source': 'test'
                })
    
    ProcessingStatus.success(f"Created test file with 100 rows")
    
    # Count lines
    with open(test_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header
    
    # Process with progress
    progress = ProgressIndicator(total_items=total_lines, show_eta=True)
    progress.start("Processing CSV file")
    
    current_user = None
    users_seen = set()
    accepted = 0
    rejected = 0
    
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader, 1):
            # Simulate processing time
            time.sleep(0.01)
            
            user_id = row['user_id']
            if user_id != current_user:
                current_user = user_id
                users_seen.add(user_id)
            
            # Randomly accept/reject
            if random.random() > 0.1:
                accepted += 1
            else:
                rejected += 1
            
            # Update progress
            progress.update(i, {
                'rows_processed': i,
                'users_processed': len(users_seen),
                'current_user': current_user,
                'accepted': accepted,
                'rejected': rejected
            })
    
    progress.finish(f"Processed {total_lines} rows from {len(users_seen)} users")
    
    # Clean up
    test_file.unlink()
    ProcessingStatus.success("Cleaned up test file")


def test_error_handling():
    """Test error scenarios."""
    ProcessingStatus.section("Testing Error Handling")
    
    progress = ProgressIndicator(total_items=50)
    progress.start("Processing with potential errors")
    
    for i in range(1, 31):
        time.sleep(0.03)
        
        # Simulate an error at item 20
        if i == 20:
            progress.error("Encountered an error at item 20")
            time.sleep(1)
            ProcessingStatus.warning("Attempting to recover...")
            time.sleep(0.5)
            ProcessingStatus.success("Recovery successful, continuing...")
            
        progress.update(i, {
            'rows_processed': i * 10,
            'errors': 1 if i >= 20 else 0
        })
    
    progress.finish("Completed with 1 error")


def main():
    """Run all progress tests."""
    print("\n" + "="*60)
    print("CLI PROGRESS INDICATION TEST SUITE")
    print("="*60)
    
    # Test each component
    test_status_messages()
    print("\n" + "-"*60 + "\n")
    
    test_progress_bar()
    print("\n" + "-"*60 + "\n")
    
    test_spinner()
    print("\n" + "-"*60 + "\n")
    
    test_real_processing()
    print("\n" + "-"*60 + "\n")
    
    test_error_handling()
    
    ProcessingStatus.section("All Tests Complete")
    ProcessingStatus.success("Progress indication system working correctly!")


if __name__ == "__main__":
    main()