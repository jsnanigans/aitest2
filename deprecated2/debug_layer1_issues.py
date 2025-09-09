#!/usr/bin/env python3
"""
Debug Layer 1 issues with specific problematic users
"""

import csv
from datetime import datetime
from pathlib import Path
import json

def analyze_user_pattern(user_id: str, limit: int = 50):
    """Analyze weight patterns for a specific user."""
    
    source_file = Path("./2025-09-05_optimized.csv")
    
    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        return
    
    measurements = []
    
    with open(source_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['user_id'] == user_id:
                measurements.append({
                    'date': row['effectivDateTime'],
                    'weight': float(row['weight']),
                    'source': row.get('source_type', 'unknown')
                })
                if len(measurements) >= limit:
                    break
    
    if not measurements:
        print(f"No data found for user {user_id}")
        return
    
    print(f"\n{'='*70}")
    print(f"User: {user_id}")
    print(f"Total measurements: {len(measurements)}")
    print(f"{'='*70}")
    
    # Analyze patterns
    weights = [m['weight'] for m in measurements]
    min_weight = min(weights)
    max_weight = max(weights)
    
    print(f"\nWeight Statistics:")
    print(f"  Min: {min_weight:.1f} kg")
    print(f"  Max: {max_weight:.1f} kg") 
    print(f"  Range: {max_weight - min_weight:.1f} kg")
    
    # Look for problematic patterns
    print(f"\nFirst 20 measurements:")
    print(f"{'Date':<25} {'Weight':>10} {'Change':>10} {'%Change':>10} {'Source':<20}")
    print("-" * 80)
    
    prev_weight = None
    prev_date = None
    issues_found = []
    
    for i, m in enumerate(measurements[:20]):
        date_str = m['date']
        weight = m['weight']
        source = m['source']
        
        change_str = ""
        pct_change_str = ""
        
        if prev_weight is not None:
            change = weight - prev_weight
            pct_change = (change / prev_weight) * 100
            change_str = f"{change:+.1f}"
            pct_change_str = f"{pct_change:+.1f}%"
            
            # Check for issues
            if abs(pct_change) > 3.0:  # More than 3% change per measurement
                issues_found.append(f"Row {i}: {pct_change:.1f}% change")
                change_str += " ⚠️"
            
            if weight < 30 or weight > 400:
                issues_found.append(f"Row {i}: Weight {weight}kg out of bounds")
                
        print(f"{date_str:<25} {weight:>10.1f} {change_str:>10} {pct_change_str:>10} {source:<20}")
        
        prev_weight = weight
        prev_date = date_str
    
    if issues_found:
        print(f"\n⚠️  Issues Found:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    # Check what Layer 1 should catch
    print(f"\n🔍 Layer 1 Analysis:")
    
    # Check if weights are within physiological bounds
    out_of_bounds = [w for w in weights if w < 30 or w > 400]
    if out_of_bounds:
        print(f"  ❌ {len(out_of_bounds)} weights outside [30, 400] kg range")
    else:
        print(f"  ✅ All weights within physiological bounds")
    
    # Check for extreme rate changes
    extreme_changes = []
    for i in range(1, len(measurements[:20])):
        prev = measurements[i-1]['weight']
        curr = measurements[i]['weight']
        pct = abs((curr - prev) / prev * 100)
        if pct > 10:  # More than 10% change
            extreme_changes.append((i, pct))
    
    if extreme_changes:
        print(f"  ❌ {len(extreme_changes)} extreme changes (>10%) detected")
        for idx, pct in extreme_changes[:5]:
            print(f"     Row {idx}: {pct:.1f}% change")
    else:
        print(f"  ✅ No extreme rate changes")
    
    return measurements


def check_layer1_logic():
    """Check the Layer 1 filter logic for bugs."""
    
    print("\n" + "="*70)
    print("LAYER 1 LOGIC CHECK")
    print("="*70)
    
    from src.filters.layer1_heuristic import StatelessRateOfChangeFilter
    
    filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
    
    # Test case: Normal sequence
    print("\nTest 1: Normal weight sequence (should pass)")
    last_state = {
        'weight': 80.0,
        'timestamp': datetime(2024, 1, 1)
    }
    
    from src.core.types import WeightMeasurement
    
    measurement = WeightMeasurement(
        user_id='test',
        weight=80.5,  # 0.6% change
        timestamp=datetime(2024, 1, 2),
        source_type='test'
    )
    
    is_valid, outlier, meta = filter.validate(measurement, last_state)
    print(f"  Weight: 80.0 -> 80.5 (0.6% change)")
    print(f"  Result: {'✅ Valid' if is_valid else '❌ Rejected'}")
    print(f"  Metadata: {meta}")
    
    # Test case: Same weight repeated (THIS MIGHT BE THE BUG!)
    print("\nTest 2: Same weight repeated (should pass)")
    last_state = {
        'weight': 80.0,
        'timestamp': datetime(2024, 1, 1)
    }
    
    measurement = WeightMeasurement(
        user_id='test',
        weight=80.0,  # Same weight
        timestamp=datetime(2024, 1, 2),
        source_type='test'
    )
    
    is_valid, outlier, meta = filter.validate(measurement, last_state)
    print(f"  Weight: 80.0 -> 80.0 (0% change)")
    print(f"  Result: {'✅ Valid' if is_valid else '❌ Rejected'}")
    print(f"  Metadata: {meta}")
    
    # Test case: Very small time delta
    print("\nTest 3: Multiple measurements same day (should handle gracefully)")
    last_state = {
        'weight': 80.0,
        'timestamp': datetime(2024, 1, 1, 8, 0)  # 8 AM
    }
    
    measurement = WeightMeasurement(
        user_id='test',
        weight=81.0,  # 1kg change
        timestamp=datetime(2024, 1, 1, 9, 0),  # 9 AM (1 hour later)
        source_type='test'
    )
    
    is_valid, outlier, meta = filter.validate(measurement, last_state)
    print(f"  Weight: 80.0 -> 81.0 (1.25% change in 1 hour)")
    print(f"  Time delta: {meta.get('time_delta_days', 0):.4f} days")
    print(f"  Result: {'✅ Valid' if is_valid else '❌ Rejected'}")
    print(f"  Metadata: {meta}")


if __name__ == "__main__":
    # Analyze the problematic users
    problematic_users = [
        "0069687c-c1b2-420e-bfae-009a284d13fe",
        "010fbe98-e372-48ec-b46b-b99093b028ad",
    ]
    
    for user_id in problematic_users:
        analyze_user_pattern(user_id)
    
    # Check Layer 1 logic
    check_layer1_logic()
    
    print("\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    print("""
Potential issues with Layer 1:
1. Rate of change calculation might be too strict for same-day measurements
2. Time delta calculation could be incorrect for small intervals
3. The filter might be rejecting valid repeated measurements
4. The percentage calculation might not handle edge cases properly
    """)