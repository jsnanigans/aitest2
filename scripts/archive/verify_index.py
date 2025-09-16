#!/usr/bin/env python3
"""
Verify the generated index.html file
"""

from pathlib import Path
import json
import re

def verify_index():
    index_path = Path("output/viz_test_no_date/index.html")
    
    if not index_path.exists():
        print("✗ Index file not found")
        return False
    
    with open(index_path, 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check for essential components
    checks.append(("HTML structure", "<html" in content and "</html>" in content))
    checks.append(("Dashboard data", "DASHBOARD_DATA" in content))
    checks.append(("JavaScript class", "class DashboardViewer" in content))
    checks.append(("User list container", 'id="userList"' in content))
    checks.append(("Dashboard container", 'id="dashboardContainer"' in content))
    checks.append(("Sort controls", 'id="sortSelect"' in content))
    checks.append(("Search box", 'id="searchBox"' in content))
    checks.append(("Keyboard hint", "arrow keys" in content.lower()))
    
    # Extract embedded data
    data_match = re.search(r'const DASHBOARD_DATA = ({.*?});', content, re.DOTALL)
    if data_match:
        try:
            data_str = data_match.group(1)
            # Basic JSON validation (not full parse due to potential escaping)
            checks.append(("Valid JSON structure", '"users"' in data_str and '"summary"' in data_str))
        except:
            checks.append(("Valid JSON structure", False))
    else:
        checks.append(("Valid JSON structure", False))
    
    # Check CSS is embedded
    checks.append(("CSS styles", "<style>" in content and "user-item" in content))
    
    # Print results
    print("Index.html Verification:")
    print("-" * 40)
    
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("-" * 40)
    
    if all_passed:
        print("✓ All checks passed!")
        
        # Additional info
        size_kb = index_path.stat().st_size / 1024
        print(f"\nFile size: {size_kb:.1f} KB")
        
        # Count users in embedded data
        user_count = content.count('"dashboard_file"')
        print(f"Users found: {user_count}")
        
    else:
        print("✗ Some checks failed")
    
    return all_passed

if __name__ == "__main__":
    verify_index()