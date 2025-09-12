"""
Investigation of why user 4d5054c4-2492-4cd8-a15c-53381fb6bd49 
has an accepted measurement of 106.14 kg on 2023-10-19.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.processor import WeightProcessor
from src.database import ProcessorStateDB

def test_106kg_issue():
    """Reproduce the exact scenario from the CSV data."""
    
    # Create database and config
    db = ProcessorStateDB()
    
    processing_config = {
        "min_weight": 30.0,
        "max_weight": 400.0,
        "max_daily_change": 0.05,
        "extreme_threshold": 0.20,
        "kalman_cleanup_threshold": 4.0,
    }
    
    kalman_config = {
        "initial_variance": 0.5,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "reset_gap_days": 30,
    }
    
    user_id = "4d5054c4-2492-4cd8-a15c-53381fb6bd49"
    
    # Process the actual measurements from CSV
    measurements = [
        (datetime(2019, 11, 12), 123.83),
        (datetime(2022, 2, 15), 136.03),  # 826 days gap - will reset
        (datetime(2022, 11, 21), 142.88),  # 279 days gap - will reset
        (datetime(2022, 12, 29), 146.51),  # 38 days gap - will reset
        (datetime(2023, 2, 16), 145.6),    # 49 days gap - will reset
        (datetime(2023, 3, 30), 148.78),   # 42 days gap - will reset
        (datetime(2023, 5, 22), 145.24),   # 53 days gap - will reset
        (datetime(2023, 7, 6), 146.51),    # 45 days gap - will reset
        (datetime(2023, 7, 13), 146.97),   # 7 days
        (datetime(2023, 9, 5), 145.15),    # 54 days gap - will reset
        (datetime(2023, 9, 11), 146.97),   # 6 days
        (datetime(2023, 10, 19), 106.14),  # 38 days gap - THIS IS THE PROBLEM!
        (datetime(2024, 2, 29), 148.33),   # 133 days gap - will reset
    ]
    
    results = []
    for timestamp, weight in measurements:
        print(f"\n{'='*60}")
        print(f"Processing: {timestamp.date()} - {weight:.2f} kg")
        
        # Check gap from last measurement
        if results:
            last_ts = results[-1]['timestamp']
            gap_days = (timestamp - last_ts).days
            print(f"Gap from last: {gap_days} days")
            if gap_days > 30:
                print(f"⚠️  GAP > 30 DAYS - SHOULD RESET!")
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        if result:
            results.append(result)
            print(f"Result: accepted={result['accepted']}")
            if 'reason' in result:
                print(f"Reason: {result['reason']}")
            if 'filtered_weight' in result:
                print(f"Filtered: {result['filtered_weight']:.2f} kg")
                
    # Analyze the 106.14 kg measurement specifically
    print(f"\n{'='*60}")
    print("ANALYSIS OF 106.14 kg MEASUREMENT:")
    print("="*60)
    
    # Find the 106.14 kg result
    for i, r in enumerate(results):
        if abs(r['raw_weight'] - 106.14) < 0.01:
            print(f"\nFound 106.14 kg measurement at index {i}:")
            print(f"  Timestamp: {r['timestamp']}")
            print(f"  Accepted: {r['accepted']}")
            if 'reason' in r:
                print(f"  Reason: {r['reason']}")
            
            # Check the gap
            if i > 0:
                prev = results[i-1]
                gap = (r['timestamp'] - prev['timestamp']).days
                print(f"\n  Gap from previous: {gap} days")
                print(f"  Previous weight: {prev['raw_weight']:.2f} kg")
                print(f"  Weight change: {r['raw_weight'] - prev['raw_weight']:.2f} kg")
                print(f"  Relative change: {abs(r['raw_weight'] - prev['raw_weight']) / prev['raw_weight'] * 100:.1f}%")
                
                if gap > 30:
                    print(f"\n  ✅ Gap > 30 days - SHOULD have reset and accepted!")
                else:
                    print(f"\n  ❌ Gap = {gap} days - Should NOT have accepted such a large deviation!")
                    print(f"     Expected behavior: Should reject as extreme outlier")
                    print(f"     Actual behavior: {'ACCEPTED' if r['accepted'] else 'REJECTED'}")

if __name__ == "__main__":
    test_106kg_issue()
