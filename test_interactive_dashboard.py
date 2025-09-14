#!/usr/bin/env python3
"""
Test script for interactive quality dashboards
Demonstrates the new Plotly-based visualization system
"""

import sys
import json
from pathlib import Path
import tomllib

def test_dashboard():
    print("=" * 60)
    print("Interactive Quality Dashboard Test")
    print("=" * 60)
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Check visualization mode
    viz_mode = config.get("visualization", {}).get("mode", "auto")
    print(f"\nVisualization Configuration:")
    print(f"  Mode: {viz_mode}")
    print(f"  Quality Scoring: {config.get('quality_scoring', {}).get('enabled', False)}")
    print(f"  Theme: {config.get('visualization', {}).get('theme', 'plotly_white')}")
    
    # Check if interactive dashboard was created
    output_dir = Path("output")
    viz_dirs = list(output_dir.glob("viz_*"))
    
    if not viz_dirs:
        print("\nNo visualization directories found. Run main.py first.")
        return
    
    latest_viz = sorted(viz_dirs)[-1]
    print(f"\nLatest visualization directory: {latest_viz}")
    
    # Find HTML dashboards
    html_files = list(latest_viz.glob("dashboard_*.html"))
    
    if html_files:
        print(f"\nFound {len(html_files)} interactive dashboard(s):")
        for html_file in html_files:
            file_size = html_file.stat().st_size / 1024  # KB
            print(f"  - {html_file.name} ({file_size:.1f} KB)")
            
            # Check if it's a valid Plotly dashboard
            with open(html_file, 'r') as f:
                content = f.read(1000)
                if 'plotly' in content.lower():
                    print("    ✓ Valid Plotly dashboard")
                else:
                    print("    ✗ Not a Plotly dashboard")
    else:
        print("\nNo interactive dashboards found.")
    
    # Check results file
    results_files = list(output_dir.glob("results_*.json"))
    if results_files:
        latest_results = sorted(results_files)[-1]
        with open(latest_results, 'r') as f:
            results = json.load(f)
        
        stats = results.get("stats", {})
        print(f"\nProcessing Statistics:")
        print(f"  Total measurements: {stats.get('accepted', 0) + stats.get('rejected', 0)}")
        print(f"  Accepted: {stats.get('accepted', 0)}")
        print(f"  Rejected: {stats.get('rejected', 0)}")
        
        # Check if quality scores are present
        users = results.get("users", {})
        has_quality = False
        for user_id, user_results in users.items():
            if user_results and user_results[0].get("quality_score") is not None:
                has_quality = True
                break
        
        if has_quality:
            print("  ✓ Quality scores are being calculated")
        else:
            print("  ✗ Quality scores not found (check if quality_scoring.enabled=true)")
    
    print("\n" + "=" * 60)
    print("Dashboard Features:")
    print("  • Interactive hover tooltips with quality component breakdown")
    print("  • Zoom and pan on time series data")
    print("  • Quality score color gradient (Red-Yellow-Green)")
    print("  • Component score breakdown chart")
    print("  • Quality distribution histogram")
    print("  • Source quality comparison sunburst")
    print("  • Rejection analysis with quality insights")
    print("  • Statistics panel with quality metrics")
    print("\nTo view the dashboard:")
    if html_files:
        print(f"  Open: {html_files[0]}")
    else:
        print("  Run: python main.py data/test_sample.csv")
    print("=" * 60)

if __name__ == "__main__":
    test_dashboard()