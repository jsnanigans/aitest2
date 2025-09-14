#!/usr/bin/env python3

import json
from pathlib import Path
from src.viz_index import create_index_from_results

def test_index_generation():
    output_dir = Path("output/viz_test_no_date")
    results_file = Path("output/results_test_no_date.json")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return
    
    print(f"Generating index from: {results_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        index_path = create_index_from_results(
            str(results_file),
            str(output_dir)
        )
        print(f"✓ Index generated successfully: {index_path}")
        
        index_file = Path(index_path)
        if index_file.exists():
            size_kb = index_file.stat().st_size / 1024
            print(f"  File size: {size_kb:.1f} KB")
            
            with open(index_file, 'r') as f:
                content = f.read()
                if 'DASHBOARD_DATA' in content:
                    print("  ✓ Dashboard data embedded")
                if 'DashboardViewer' in content:
                    print("  ✓ JavaScript included")
                if 'user-list' in content:
                    print("  ✓ HTML structure present")
        
    except Exception as e:
        print(f"✗ Error generating index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_index_generation()