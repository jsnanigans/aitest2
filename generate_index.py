#!/usr/bin/env python3
"""
Generate index.html for existing visualization directories
"""

import argparse
import sys
from pathlib import Path
from src.viz_index import create_index_from_results

def main():
    parser = argparse.ArgumentParser(
        description="Generate index.html for dashboard navigation"
    )
    parser.add_argument(
        "results_file",
        help="Path to results JSON file (e.g., output/results_test.json)"
    )
    parser.add_argument(
        "--viz-dir",
        help="Visualization directory (defaults to sibling of results file)"
    )
    parser.add_argument(
        "--output",
        default="index.html",
        help="Output filename (default: index.html)"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    if args.viz_dir:
        viz_dir = Path(args.viz_dir)
    else:
        results_name = results_path.stem
        viz_name = results_name.replace("results_", "viz_")
        viz_dir = results_path.parent / viz_name
    
    if not viz_dir.exists():
        print(f"Error: Visualization directory not found: {viz_dir}")
        print(f"Tip: Specify the directory with --viz-dir")
        sys.exit(1)
    
    print(f"Generating index from: {results_path}")
    print(f"Visualization directory: {viz_dir}")
    
    try:
        index_path = create_index_from_results(
            str(results_path),
            str(viz_dir),
            args.output
        )
        print(f"✓ Index generated: {index_path}")
        
        index_file = Path(index_path)
        size_kb = index_file.stat().st_size / 1024
        print(f"  File size: {size_kb:.1f} KB")
        
        with open(results_path) as f:
            import json
            data = json.load(f)
            user_count = len(data.get("users", {}))
            print(f"  Users indexed: {user_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()