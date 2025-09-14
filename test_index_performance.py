#!/usr/bin/env python3
"""
Test the performance improvements in the index.html
"""

from pathlib import Path
import re

def analyze_index_performance():
    index_path = Path("output_demo/viz_test_no_date/index.html")
    
    if not index_path.exists():
        print("✗ Index file not found")
        return False
    
    with open(index_path, 'r') as f:
        content = f.read()
    
    print("Index.html Performance Analysis:")
    print("=" * 50)
    
    # Check for performance improvements
    checks = []
    
    # 1. Check for debounced search
    if "searchTimer" in content and "setTimeout" in content and "150" in content:
        checks.append(("✓", "Search input debounced (150ms)"))
    else:
        checks.append(("✗", "Search input NOT debounced"))
    
    # 2. Check for duplicate iframe prevention
    if "loadingIframe" in content and "if (this.loadingIframe) return" in content:
        checks.append(("✓", "Iframe loading guard present"))
    else:
        checks.append(("✗", "No iframe loading guard"))
    
    # 3. Check for load state tracking
    if "loadHandled" in content:
        checks.append(("✓", "Load state tracking implemented"))
    else:
        checks.append(("✗", "No load state tracking"))
    
    # 4. Check for requestAnimationFrame usage
    raf_count = content.count("requestAnimationFrame")
    if raf_count > 0:
        checks.append(("✓", f"Using requestAnimationFrame ({raf_count} times)"))
    else:
        checks.append(("✗", "Not using requestAnimationFrame"))
    
    # 5. Check for proper timeout (3000ms instead of 100ms)
    if "3000" in content and "setTimeout" in content:
        checks.append(("✓", "Timeout increased to 3000ms"))
    else:
        checks.append(("✗", "Still using short timeout"))
    
    # 6. Check for duplicate selection prevention
    if "if (this.currentUser === user) return" in content:
        checks.append(("✓", "Duplicate selection prevention"))
    else:
        checks.append(("✗", "No duplicate selection check"))
    
    # 7. Check for existing iframe check
    if "existingIframe" in content and "endsWith(user.dashboard_file)" in content:
        checks.append(("✓", "Checks for existing iframe"))
    else:
        checks.append(("✗", "No existing iframe check"))
    
    # 8. Check for optimized scrolling
    if "itemRect.top < containerRect.top" in content:
        checks.append(("✓", "Optimized scroll logic"))
    else:
        checks.append(("✗", "Basic scroll logic"))
    
    # 9. Check CSS performance hints
    if "will-change" in content:
        checks.append(("✓", "CSS will-change hints present"))
    else:
        checks.append(("✗", "No CSS performance hints"))
    
    # 10. Check transition optimization
    if "transition: background-color" in content or "transition: all 0.2s" not in content:
        checks.append(("✓", "Optimized CSS transitions"))
    else:
        checks.append(("✗", "Using transition: all"))
    
    # Print results
    passed = 0
    for status, description in checks:
        print(f"{status} {description}")
        if status == "✓":
            passed += 1
    
    print("=" * 50)
    print(f"Performance Score: {passed}/{len(checks)}")
    
    if passed == len(checks):
        print("✓ All performance optimizations implemented!")
    elif passed >= len(checks) * 0.8:
        print("✓ Most performance optimizations implemented")
    else:
        print("⚠ Some performance optimizations missing")
    
    # Additional metrics
    print("\nAdditional Metrics:")
    print(f"  File size: {index_path.stat().st_size / 1024:.1f} KB")
    
    # Count DOM operations
    dom_ops = content.count("innerHTML")
    print(f"  innerHTML operations: {dom_ops}")
    
    # Check for event delegation vs individual listeners
    click_listeners = content.count("addEventListener('click'")
    print(f"  Click listeners: {click_listeners}")
    
    return passed == len(checks)

if __name__ == "__main__":
    analyze_index_performance()