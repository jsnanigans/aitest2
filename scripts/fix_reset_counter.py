#!/usr/bin/env python3
"""
Fix the measurements_since_reset counter issue in processor.py
"""

import re

def fix_processor():
    # Read the current processor.py
    with open('src/processor.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Add counter increment in the main acceptance path (around line 451)
    # Find the pattern where we save state after accepting
    pattern1 = r"(    # Save updated state\n    state\['last_source'\] = source)"
    replacement1 = r"    # Save updated state\n    # Increment measurements counter for adaptation tracking\n    state['measurements_since_reset'] = state.get('measurements_since_reset', 0) + 1\n    state['last_source'] = source"
    
    content = re.sub(pattern1, replacement1, content)
    
    # Fix 2: Also ensure the counter is properly preserved during reset
    # The reset creates a new state with measurements_since_reset: 0
    # which is correct, we just need to make sure it's incremented properly
    
    # Write the fixed content
    with open('src/processor.py', 'w') as f:
        f.write(content)
    
    print("Fixed processor.py:")
    print("1. Added measurements_since_reset increment in main acceptance path")
    print("2. Counter will now properly track measurements after reset")

if __name__ == "__main__":
    fix_processor()
