#!/usr/bin/env python3
"""
Batch visualization script for processing multiple users with enhanced dashboards.
"""

import json
import sys
from pathlib import Path
import tomllib
import traceback

def batch_visualize(results_file: str, max_users: int = 10):
    """Process multiple users and generate enhanced dashboards."""
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    users = data.get("users", {})
    total_users = len(users)
    
    print(f"Found {total_users} users in results file")
    
    # Determine output directory
    output_dir = Path("output/batch_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process users
    successful = 0
    failed = 0
    
    for idx, (user_id, results) in enumerate(users.items(), 1):
        if idx > max_users:
            print(f"Reached max_users limit ({max_users})")
            break
        
        print(f"[{idx}/{min(total_users, max_users)}] Processing {user_id[:8]}...", end=" ")
        
        try:
            # Try enhanced dashboard first
            from src.viz_plotly_enhanced import create_enhanced_dashboard
            output_path = create_enhanced_dashboard(results, user_id, str(output_dir), config)
            
            if output_path:
                print("✓ Enhanced")
                successful += 1
            else:
                # Fall back to standard dashboard
                from src.viz_plotly import create_interactive_dashboard
                output_path = create_interactive_dashboard(results, user_id, str(output_dir), config)
                if output_path:
                    print("✓ Standard")
                    successful += 1
                else:
                    print("✗ Failed")
                    failed += 1
                    
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed += 1
            if "--debug" in sys.argv:
                traceback.print_exc()
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    
    # List generated files
    html_files = list(output_dir.glob("*.html"))
    if html_files:
        print(f"\nGenerated {len(html_files)} dashboard files:")
        for f in html_files[:5]:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
        if len(html_files) > 5:
            print(f"  ... and {len(html_files) - 5} more")

if __name__ == "__main__":
    # Find latest results file
    output_dir = Path("output")
    results_files = list(output_dir.glob("results_*.json"))
    
    if not results_files:
        print("No results files found. Run main.py first.")
        sys.exit(1)
    
    latest_results = sorted(results_files)[-1]
    print(f"Using results from: {latest_results}")
    
    # Get max users from command line or use default
    max_users = 10
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_users = int(sys.argv[1])
    
    batch_visualize(str(latest_results), max_users)