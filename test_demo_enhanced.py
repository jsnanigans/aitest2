#!/usr/bin/env python3
"""
Test enhanced visualization with demo data.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.viz_enhanced import create_enhanced_weight_timeline


def main():
    # Load demo results
    results_file = "demo_enhanced_output/results_demo.json"
    
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    users = results_data.get('users', {})
    
    for user_id, measurements in users.items():
        if not measurements:
            continue
        
        print(f"Creating enhanced visualization for {user_id}")
        print(f"  Total measurements: {len(measurements)}")
        
        accepted = [m for m in measurements if m.get('accepted', False)]
        rejected = [m for m in measurements if not m.get('accepted', False)]
        
        print(f"  Accepted: {len(accepted)}")
        print(f"  Rejected: {len(rejected)}")
        
        # Check data quality
        if accepted:
            sample = accepted[0]
            print(f"  Sample accepted measurement keys: {list(sample.keys())[:10]}")
        
        if rejected:
            sample = rejected[0]
            print(f"  Sample rejected measurement keys: {list(sample.keys())[:10]}")
            if 'quality_components' in sample:
                print(f"  Quality components: {sample['quality_components']}")
        
        # Create enhanced visualization
        try:
            html_path = create_enhanced_weight_timeline(
                measurements,
                user_id,
                output_dir="demo_enhanced_output"
            )
            print(f"  ✓ Created: {html_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()