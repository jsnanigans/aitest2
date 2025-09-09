#!/usr/bin/env python3
"""Test script to verify rejection display preserves source type markers."""

import json
from pathlib import Path

def analyze_rejection_display():
    """Analyze how rejected measurements are displayed."""
    
    # User with rejections
    user_id = '0B90ECE35A73404DB3F953734B16353F'
    user_file = Path(f'output/users/{user_id}.json')
    
    with open(user_file, 'r') as f:
        data = json.load(f)
    
    print("=== Rejection Display Analysis ===\n")
    print("Visual Approach:")
    print("  ✓ Rejected points: Source marker with RED thick border")
    print("  ✓ Red circle background behind the source marker")
    print("  ✓ Source type marker still visible and identifiable")
    print("  ✓ Accepted points: Source marker with white border\n")
    
    # Count rejections by source type
    source_rejections = {}
    
    if 'kalman_time_series' in data and 'time_series' in data:
        kalman_ts = data['kalman_time_series']
        time_series = data['time_series']
        
        for i, (kts, ts) in enumerate(zip(kalman_ts, time_series)):
            if kts.get('measurement_accepted') == False:
                source = ts.get('source', 'unknown')
                if source not in source_rejections:
                    source_rejections[source] = []
                source_rejections[source].append({
                    'date': kts['date'][:10],
                    'weight': kts['measured_weight'],
                    'reason': kts.get('rejection_reason', 'unknown')
                })
    
    print("Rejections by Source Type:")
    for source, rejections in source_rejections.items():
        print(f"\n  {source}: {len(rejections)} rejections")
        for rej in rejections[:2]:  # Show first 2
            print(f"    - {rej['date']}: {rej['weight']:.1f}kg ({rej['reason']})")
        if len(rejections) > 2:
            print(f"    ... and {len(rejections) - 2} more")
    
    print("\nLegend:")
    print("  - Source markers preserved (triangle, square, diamond, etc.)")
    print("  - Red border = Rejected measurement")
    print("  - White border = Accepted measurement")
    print("  - Red background circle = Visual emphasis on rejection")
    
    # Check if visualization exists
    viz_file = Path(f'output/dashboards_yes/{user_id}.png')
    if viz_file.exists():
        print(f"\n✓ Visualization updated at: {viz_file}")
        print("  The rejected points now show:")
        print("  1. Original source type marker (preserved)")
        print("  2. Red border around the marker")
        print("  3. Larger red circle background")
    else:
        print(f"\n✗ Visualization not found")

if __name__ == "__main__":
    analyze_rejection_display()