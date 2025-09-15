"""
Update plausibility scoring to use MAD (Median Absolute Deviation)
as recommended in the research for robustness.
"""

# Read the current file
with open('src/quality_scorer.py', 'r') as f:
    lines = f.readlines()

# Find and replace the calculate_plausibility_score method
new_plausibility = '''    def calculate_plausibility_score(
        self,
        weight: float,
        recent_weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate plausibility based on statistical deviation.
        Uses MAD (Median Absolute Deviation) for robustness as recommended in research.
        Stateless - all context from parameters.
        
        Args:
            weight: Current weight measurement
            recent_weights: List of recent accepted weights from state
            
        Returns:
            Score from 0.0 (implausible) to 1.0 (plausible)
        """
        if not recent_weights or len(recent_weights) < 3:
            return 0.8
        
        # Use recent history window
        recent_array = np.array(recent_weights[-20:])
        median_weight = np.median(recent_array)
        
        # Calculate MAD for robust variance estimation
        mad = np.median(np.abs(recent_array - median_weight))
        
        # Handle zero MAD (all weights identical)
        if mad < 0.1:
            mad = 0.5
        
        # Scale factor for normal distribution
        robust_std = 1.4826 * mad
        
        # Calculate deviation
        deviation = abs(weight - median_weight)
        z_score = deviation / robust_std
        
        # Also check percentage deviation
        deviation_percent = (deviation / median_weight) * 100
        
        # Score based on both z-score and percentage
        if deviation_percent <= 2.0 and z_score <= 1.5:
            return 1.0
        elif deviation_percent <= 3.0 and z_score <= 2.5:
            return 0.9
        elif deviation_percent <= 5.0 and z_score <= 3.5:
            return 0.7
        else:
            # Exponential decay for extreme values
            score = np.exp(-0.5 * (z_score - 3))
            return max(0.0, min(0.5, score))
    '''

# Find the method start
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if 'def calculate_plausibility_score(' in line:
        start_idx = i
    if start_idx is not None and i > start_idx:
        if line.strip() and not line[0].isspace() and 'def ' in line:
            end_idx = i
            break

# Replace the method
if start_idx is not None and end_idx is not None:
    new_lines = lines[:start_idx] + [new_plausibility + '\n'] + lines[end_idx:]
    
    with open('src/quality_scorer.py', 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated calculate_plausibility_score method (lines {start_idx+1} to {end_idx})")
else:
    print("Could not find method to replace")
