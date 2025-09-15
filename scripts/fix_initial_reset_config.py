#!/usr/bin/env python3
"""
Fix the initial reset configuration to properly use config values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_reset_manager():
    """Update reset_manager.py to have better defaults for initial reset."""
    
    file_path = "src/reset_manager.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and update the INITIAL reset defaults
    for i, line in enumerate(lines):
        if "ResetType.INITIAL: {" in line:
            # Update the defaults for INITIAL reset to be more aggressive
            lines[i+1] = "                'weight_boost_factor': 20,\n"  # was 10
            lines[i+2] = "                'trend_boost_factor': 200,\n"  # was 100
            lines[i+3] = "                'decay_rate': 2,\n"  # was 3 (lower = faster)
            lines[i+4] = "                'warmup_measurements': 15,\n"  # was 10
            lines[i+5] = "                'adaptive_days': 10,\n"  # was 7
            lines[i+6] = "                'quality_threshold': 0.3,\n"  # was 0.4 (more lenient)
            break
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated {file_path} with better INITIAL reset defaults")

def fix_kalman_adaptive():
    """Update kalman_adaptive.py to properly boost initial variance."""
    
    file_path = "src/kalman_adaptive.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the hardcoded initial_variance
    content = content.replace(
        "        adaptive_params = {\n            'initial_variance': 5.0,",
        "        # Boost initial variance based on reset type\n"
        "        initial_variance_boost = reset_params.get('initial_variance_boost', 3.0)\n"
        "        adaptive_params = {\n            'initial_variance': base_config.get('initial_variance', 0.361) * initial_variance_boost,"
    )
    
    # Fix observation covariance to be adaptive
    content = content.replace(
        "            'observation_covariance': 2.0,",
        "            'observation_covariance': base_config.get('observation_covariance', 3.4) * 0.5,  # More lenient during adaptation"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path} to properly boost initial variance")

def fix_processor_adaptive_check():
    """Update processor.py to use reset parameters for adaptive period."""
    
    file_path = "src/processor.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and fix the hardcoded adaptive period check
    for i, line in enumerate(lines):
        if "measurements_since_reset < 10:" in line:
            # Get reset parameters from state
            new_lines = [
                "            # Get adaptive parameters from reset state\n",
                "            reset_params = state.get('reset_parameters', {})\n",
                "            warmup_measurements = reset_params.get('warmup_measurements', 10)\n",
                "            if measurements_since_reset < warmup_measurements:\n"
            ]
            lines[i] = ''.join(new_lines)
        
        elif "days_since < 7:" in line:
            # Use adaptive_days from reset parameters
            lines[i] = "                    adaptive_days = reset_params.get('adaptive_days', 7)\n"
            lines.insert(i+1, "                    if days_since < adaptive_days:\n")
            del lines[i+2]  # Remove old line
        
        elif 'adaptive_quality_config["threshold"] = 0.4' in line:
            # Use quality_threshold from reset parameters
            new_lines = [
                "            # Use threshold from reset parameters if available\n",
                "            reset_params = state.get('reset_parameters', {})\n",
                "            adaptive_threshold = reset_params.get('quality_threshold', 0.4)\n",
                "            adaptive_quality_config['threshold'] = adaptive_threshold\n"
            ]
            lines[i] = ''.join(new_lines)
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated {file_path} to use reset parameters for adaptive period")

def update_config():
    """Update config.toml with better initial reset parameters."""
    
    config_path = "config.toml"
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    in_initial_section = False
    for i, line in enumerate(lines):
        if "[kalman.reset.initial]" in line:
            in_initial_section = True
        elif line.startswith("[") and in_initial_section:
            in_initial_section = False
        
        if in_initial_section:
            # Update specific parameters
            if "weight_boost_factor" in line:
                lines[i] = "weight_boost_factor = 30  # Very aggressive for initial (was 20)\n"
            elif "trend_boost_factor" in line:
                lines[i] = "trend_boost_factor = 200  # Very aggressive for initial (was 100)\n"
            elif "decay_rate" in line:
                lines[i] = "decay_rate = 2  # Faster decay for quicker adaptation (was 3)\n"
            elif "warmup_measurements" in line:
                lines[i] = "warmup_measurements = 15  # More measurements for initial learning (was 10)\n"
            elif "adaptive_days" in line:
                lines[i] = "adaptive_days = 14  # Two weeks for initial adaptation (was 10)\n"
            elif "quality_threshold" in line:
                lines[i] = "quality_threshold = 0.3  # Very lenient for initial (was 0.4)\n"
    
    # Add new parameters if not present
    if in_initial_section:
        # Find where to insert
        for i, line in enumerate(lines):
            if "[kalman.reset.initial]" in line:
                # Find the end of this section
                j = i + 1
                while j < len(lines) and not lines[j].startswith("["):
                    j += 1
                # Insert new parameters before the next section
                lines.insert(j, "initial_variance_boost = 5  # Boost initial variance for faster adaptation\n")
                lines.insert(j+1, "quality_plausibility_weight = 0.05  # Very low during initial\n")
                lines.insert(j+2, "quality_consistency_weight = 0.10  # Low during initial\n")
                lines.insert(j+3, "\n")
                break
    
    with open(config_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated {config_path} with better initial reset configuration")

if __name__ == "__main__":
    print("Fixing initial reset configuration...")
    print()
    
    fix_reset_manager()
    fix_kalman_adaptive()
    fix_processor_adaptive_check()
    update_config()
    
    print()
    print("Done! The initial reset should now be more adaptive.")
    print()
    print("Key improvements:")
    print("1. Initial reset now uses more aggressive boost factors (30x weight, 200x trend)")
    print("2. Quality threshold lowered to 0.3 for initial measurements")
    print("3. Adaptive period extended to 15 measurements / 14 days")
    print("4. Initial variance properly boosted (5x)")
    print("5. All parameters now properly read from config instead of hardcoded")
