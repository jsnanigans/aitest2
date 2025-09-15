#!/usr/bin/env python3
"""
Fix the adaptive reset implementation to work correctly.
"""

import re

def fix_implementation():
    # Read the processor file
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'r') as f:
        content = f.read()
    
    # Remove the misplaced adaptive covariance code (lines 173-184)
    lines = content.split('\n')
    
    # Find and remove the misplaced block
    new_lines = []
    skip_until = -1
    for i, line in enumerate(lines):
        if i < skip_until:
            continue
        if '# Use adaptive covariances if recently reset' in line and i < 200:
            # Skip this block (it's in the wrong place)
            skip_until = i + 12  # Skip the next 11 lines
            continue
        new_lines.append(line)
    
    # Join back
    content = '\n'.join(new_lines)
    
    # Now add the measurement increment in the right place
    # It should be after an accepted measurement, near line 409
    if 'state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1' not in content:
        # Add it before db.save_state in the main flow
        content = content.replace(
            '    state[\'last_accepted_timestamp\'] = timestamp\n    db.save_state(user_id, state)',
            '    state[\'last_accepted_timestamp\'] = timestamp\n    # Increment measurements since reset\n    state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1\n    db.save_state(user_id, state)'
        )
    
    # Write back
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed processor.py implementation")
    
    # Now ensure the config has the right settings
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/config.toml', 'r') as f:
        config = f.read()
    
    if '[kalman.post_reset_adaptation]' not in config:
        # Add after [kalman.reset]
        config = config.replace(
            '[kalman.reset]\n# Simple reset after long gaps (replaces complex gap handling)\nenabled = true  # Enable automatic reset after long gaps\ngap_threshold_days = 30  # Days without data before full reset',
            '[kalman.reset]\n# Simple reset after long gaps (replaces complex gap handling)\nenabled = true  # Enable automatic reset after long gaps\ngap_threshold_days = 30  # Days without data before full reset\n\n[kalman.post_reset_adaptation]\n# Adaptive parameters for quick adaptation after reset\nenabled = true  # Enable adaptive covariances after reset\nwarmup_measurements = 10  # Number of measurements to adapt over\nweight_boost_factor = 10  # Initial multiplier for weight covariance\ntrend_boost_factor = 100  # Initial multiplier for trend covariance\ndecay_rate = 3  # Exponential decay rate (smaller = faster decay)'
        )
        
        with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/config.toml', 'w') as f:
            f.write(config)
        print("✓ Updated config.toml with adaptive settings")
    
    print("\nImplementation fixed!")

if __name__ == "__main__":
    fix_implementation()
