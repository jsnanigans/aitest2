#!/usr/bin/env python3
"""Test script to verify diagnostic dashboard layout improvements."""

import json
from pathlib import Path
from src.visualization import DiagnosticDashboard

def test_layout_improvements():
    """Test the diagnostic dashboard with layout improvements."""
    
    # Load existing results
    results_file = Path('output/results_test_no_date.json')
    if not results_file.exists():
        print("No test data found. Please run main.py first.")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'users' not in data:
        print("No user data found in results.")
        return
    
    users_data = data['users']
    
    # Find users with different amounts of data
    test_users = []
    for user_id, measurements in users_data.items():
        if len(measurements) > 50:
            test_users.append((user_id, len(measurements), "large"))
        elif len(measurements) > 20:
            test_users.append((user_id, len(measurements), "medium"))
        elif len(measurements) > 5:
            test_users.append((user_id, len(measurements), "small"))
        
        if len(test_users) >= 3:
            break
    
    if not test_users:
        print("No suitable test users found.")
        return
    
    print("Testing diagnostic dashboard layout improvements...")
    print("=" * 60)
    
    for user_id, count, size in test_users:
        print(f"\nProcessing {size} dataset: User {user_id[:8]}... ({count} measurements)")
        
        try:
            # Create diagnostic dashboard
            dashboard = DiagnosticDashboard()
            output_path = dashboard.create_dashboard(
                users_data[user_id], 
                user_id, 
                output_dir='test_layout_fix'
            )
            print(f"  ✓ Dashboard created: {output_path}")
            
            # Verify file was created and has reasonable size
            file_size = Path(output_path).stat().st_size
            print(f"  ✓ File size: {file_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Layout improvements applied:")
    print("  • Increased vertical spacing: 0.08 → 0.15")
    print("  • Increased horizontal spacing: 0.12 → 0.18")
    print("  • Increased figure height: 1400 → 1800 pixels")
    print("  • Added row height ratios for better space distribution")
    print("  • Fixed legend positioning to prevent overlays")
    print("  • Reduced subplot title font size to 12px")
    print("  • Adjusted marker sizes and opacity for clarity")
    print("  • Added margin adjustments for better layout")
    print("\nDashboards saved to: test_layout_fix/")
    print("\nOpen the HTML files in a browser to verify:")
    print("  1. No overlapping chart titles")
    print("  2. Proper spacing between subplots")
    print("  3. Legend doesn't overlay on chart content")
    print("  4. All text is readable")
    print("  5. Summary table is properly formatted")

if __name__ == "__main__":
    test_layout_improvements()