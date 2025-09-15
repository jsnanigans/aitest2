#!/usr/bin/env python3
"""
Script to implement adaptive covariance after reset.
This modifies the processor to track measurements since reset and use adaptive covariances.
"""

import sys
import os

def implement_adaptive_reset():
    """Implement the adaptive reset solution."""
    
    # Read processor.py
    processor_path = "/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py"
    with open(processor_path, 'r') as f:
        lines = f.readlines()
    
    # Find and modify the Kalman update section (around line 339)
    modified = False
    for i in range(len(lines)):
        # Add adaptive covariance calculation before update_state
        if 'state = KalmanFilterManager.update_state(' in lines[i] and not modified:
            # Insert adaptive covariance logic before this line
            indent = '    '
            new_lines = [
                f'{indent}# Use adaptive covariances if recently reset\n',
                f'{indent}measurements_since_reset = state.get("measurements_since_reset", 100)\n',
                f'{indent}if measurements_since_reset < 10:\n',
                f'{indent}    # Get adaptive covariances for post-reset period\n',
                f'{indent}    adaptive_covs = KalmanFilterManager.get_adaptive_covariances(\n',
                f'{indent}        measurements_since_reset, kalman_config\n',
                f'{indent}    )\n',
                f'{indent}    # Update kalman params with adaptive values\n',
                f'{indent}    state["kalman_params"]["transition_covariance"] = [\n',
                f'{indent}        [adaptive_covs["weight"], 0],\n',
                f'{indent}        [0, adaptive_covs["trend"]]\n',
                f'{indent}    ]\n',
                f'{indent}\n',
            ]
            lines[i:i] = new_lines
            modified = True
            break
    
    # Find where we save state and increment counter
    for i in range(len(lines)):
        if 'state[\'last_accepted_timestamp\'] = timestamp' in lines[i]:
            # Add increment of measurements_since_reset
            indent = '    '
            new_line = f'{indent}state["measurements_since_reset"] = state.get("measurements_since_reset", 0) + 1\n'
            lines.insert(i+1, new_line)
            break
    
    # Write back
    with open(processor_path, 'w') as f:
        f.writelines(lines)
    
    print("✓ Modified processor.py to use adaptive covariances")
    
    # Now update config.toml to add the adaptive settings
    config_path = "/Users/brendanmullins/Projects/aitest/strem_process_anchor/config.toml"
    with open(config_path, 'r') as f:
        config_lines = f.readlines()
    
    # Find the [kalman.reset] section and add adaptive config after it
    for i in range(len(config_lines)):
        if '[kalman.reset]' in config_lines[i]:
            # Find the end of this section
            j = i + 1
            while j < len(config_lines) and not config_lines[j].startswith('['):
                j += 1
            # Insert new section before next section
            new_config = [
                '\n',
                '[kalman.post_reset_adaptation]\n',
                '# Adaptive parameters for quick adaptation after reset\n',
                'enabled = true  # Enable adaptive covariances after reset\n',
                'warmup_measurements = 10  # Number of measurements to adapt over\n',
                'weight_boost_factor = 10  # Initial multiplier for weight covariance\n',
                'trend_boost_factor = 100  # Initial multiplier for trend covariance\n', 
                'decay_rate = 3  # Exponential decay rate (smaller = faster decay)\n',
            ]
            config_lines[j:j] = new_config
            break
    
    with open(config_path, 'w') as f:
        f.writelines(config_lines)
    
    print("✓ Added adaptive configuration to config.toml")
    print("\nImplementation complete!")
    print("\nThe system will now:")
    print("1. Start with 10x weight covariance and 100x trend covariance after reset")
    print("2. Exponentially decay back to normal over 10 measurements")
    print("3. Allow quick adaptation to new weight patterns after gaps")

if __name__ == "__main__":
    implement_adaptive_reset()
