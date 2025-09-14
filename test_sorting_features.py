#!/usr/bin/env python3
"""
Test the enhanced sorting features in index.html
"""

from pathlib import Path
import re

def test_sorting_features():
    index_path = Path("output_demo/viz_test_no_date/index.html")
    
    if not index_path.exists():
        print("✗ Index file not found")
        return False
    
    with open(index_path, 'r') as f:
        content = f.read()
    
    print("Index.html Sorting Features Test:")
    print("=" * 50)
    
    # Check for sorting options
    checks = []
    
    # 1. Check for all sort options
    sort_options = [
        "User ID",
        "Total Measurements", 
        "Accepted Count",
        "Rejected Count",
        "Acceptance Rate (%)",
        "First Date",
        "Last Date"
    ]
    
    for option in sort_options:
        if option in content:
            checks.append(("✓", f"Sort option: {option}"))
        else:
            checks.append(("✗", f"Missing sort option: {option}"))
    
    # 2. Check for quick sort buttons
    if "quick-sort-btn" in content:
        checks.append(("✓", "Quick sort buttons present"))
    else:
        checks.append(("✗", "No quick sort buttons"))
    
    if "Most Rejected" in content:
        checks.append(("✓", "Most Rejected quick sort"))
    else:
        checks.append(("✗", "Missing Most Rejected button"))
    
    if "Lowest Rate" in content:
        checks.append(("✓", "Lowest Rate quick sort"))
    else:
        checks.append(("✗", "Missing Lowest Rate button"))
    
    if "Most Data" in content:
        checks.append(("✓", "Most Data quick sort"))
    else:
        checks.append(("✗", "Missing Most Data button"))
    
    # 3. Check for visual indicators
    if "stat-sorted" in content:
        checks.append(("✓", "Visual sorting indicators"))
    else:
        checks.append(("✗", "No visual sorting indicators"))
    
    # 4. Check for sort label
    if "Sort by:" in content:
        checks.append(("✓", "Sort label present"))
    else:
        checks.append(("✗", "No sort label"))
    
    # 5. Check for enhanced styling
    if "sort-select:hover" in content or "border-color: #2196f3" in content:
        checks.append(("✓", "Enhanced sort styling"))
    else:
        checks.append(("✗", "Basic sort styling only"))
    
    # 6. Check for date sorting logic
    if "first_date" in content and "last_date" in content:
        checks.append(("✓", "Date sorting logic present"))
    else:
        checks.append(("✗", "No date sorting logic"))
    
    # 7. Check for sort direction button
    if 'title="Click to reverse sort"' in content:
        checks.append(("✓", "Sort direction tooltip"))
    else:
        checks.append(("✗", "No sort direction tooltip"))
    
    # Print results
    passed = 0
    for status, description in checks:
        print(f"{status} {description}")
        if status == "✓":
            passed += 1
    
    print("=" * 50)
    print(f"Sorting Features Score: {passed}/{len(checks)}")
    
    if passed == len(checks):
        print("✓ All sorting features implemented!")
    elif passed >= len(checks) * 0.8:
        print("✓ Most sorting features implemented")
    else:
        print("⚠ Some sorting features missing")
    
    # Additional info
    print("\nSorting Capabilities:")
    print("  • Sort by 7 different fields")
    print("  • Quick sort buttons for common scenarios")
    print("  • Visual indicators for sorted field")
    print("  • Ascending/descending toggle")
    print("  • Improved UI with hover effects")
    
    return passed == len(checks)

if __name__ == "__main__":
    test_sorting_features()