#!/usr/bin/env python3
"""
Fix the initial Kalman initialization to use adaptive parameters.
"""

def fix_initial_adaptive():
    # Read processor.py
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'r') as f:
        lines = f.readlines()
    
    # Find the initialization section (around line 152-159)
    for i in range(len(lines)):
        if 'if not state.get(\'kalman_params\'):' in lines[i]:
            # Found the initialization block
            # Look for the adaptive_kalman_config line
            for j in range(i, min(i+20, len(lines))):
                if 'adaptive_kalman_config = get_adaptive_kalman_params(' in lines[j]:
                    # Replace the logic to ALWAYS use adaptive for initial
                    # Change from:
                    # reset_timestamp = get_reset_timestamp(state) if reset_occurred else None
                    # To:
                    # reset_timestamp = get_reset_timestamp(state) if reset_occurred else timestamp
                    
                    # Find the reset_timestamp line
                    for k in range(j-5, j):
                        if 'reset_timestamp = get_reset_timestamp(state) if reset_occurred else None' in lines[k]:
                            lines[k] = '        # For initial measurements, treat current timestamp as "reset" to get adaptive params\n'
                            lines[k] += '        reset_timestamp = get_reset_timestamp(state) if reset_occurred else timestamp\n'
                            print(f"✓ Fixed line {k+1}: Now uses adaptive params for initial measurements")
                            break
                    break
            break
    
    # Write back
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'w') as f:
        f.writelines(lines)
    
    print("✓ Initial measurements will now use adaptive parameters")
    print("\nThis means:")
    print("- First measurement initializes with loose covariances")
    print("- Kalman can quickly adapt to actual weight level")
    print("- Prevents rejection of valid initial measurements")

if __name__ == "__main__":
    fix_initial_adaptive()
