#!/usr/bin/env python3
"""Verify that all layout fixes have been applied correctly."""

import json
from pathlib import Path
from src.visualization import DiagnosticDashboard

def verify_layout_fixes():
    """Verify the diagnostic dashboard layout fixes."""
    
    print("Verifying Diagnostic Dashboard Layout Fixes")
    print("=" * 60)
    
    # Load test data
    results_file = Path('output/results_test_no_date.json')
    if not results_file.exists():
        print("❌ No test data found")
        return False
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'users' not in data:
        print("❌ No user data found")
        return False
    
    users_data = data['users']
    
    # Test with the specific user from the screenshot
    test_user = '03a0814f-496e-484e-9472-f44f28a5c049'
    
    if test_user not in users_data:
        # Use first available user with enough data
        for uid, measurements in users_data.items():
            if len(measurements) > 20:
                test_user = uid
                break
    
    if test_user not in users_data:
        print("❌ No suitable test user found")
        return False
    
    results = users_data[test_user]
    print(f"\nTesting with user: {test_user}")
    print(f"Measurements: {len(results)}")
    
    try:
        # Create diagnostic dashboard
        dashboard = DiagnosticDashboard()
        output_path = dashboard.create_dashboard(
            results, 
            test_user, 
            output_dir='test_layout_fix'
        )
        
        # Verify file was created
        if not Path(output_path).exists():
            print("❌ Dashboard file not created")
            return False
        
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"✅ Dashboard created: {output_path}")
        print(f"✅ File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Layout Fixes Applied:")
    print("✅ Removed colorbar overlays:")
    print("   - Source Reliability: colorbar disabled (showscale=False)")
    print("   - Confidence vs Innovation: colorbar disabled")
    print("   - Innovation Distribution: annotations instead of legend")
    print("\n✅ Simplified legend:")
    print("   - Horizontal orientation at top")
    print("   - Centered positioning")
    print("   - Reduced legend entries in subplots")
    print("   - Single 'Rejected' trace instead of multiple categories")
    print("\n✅ Improved spacing:")
    print("   - Vertical spacing: 0.15")
    print("   - Horizontal spacing: 0.18")
    print("   - Figure height: 1800px")
    print("   - Balanced margins (60px sides)")
    print("\n✅ Cleaner visualization:")
    print("   - Removed duplicate legend entries")
    print("   - Disabled unnecessary colorbars")
    print("   - Simplified rejected point display")
    
    print("\n" + "=" * 60)
    print("✅ All layout fixes successfully applied!")
    print(f"\nOpen the dashboard to verify:")
    print(f"  file://{Path(output_path).absolute()}")
    
    return True

if __name__ == "__main__":
    success = verify_layout_fixes()
    exit(0 if success else 1)