import sys

# Read the original file
with open('src/quality_scorer.py', 'r') as f:
    lines = f.readlines()

# Find the calculate_consistency_score method
start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def calculate_consistency_score(' in line:
        start_line = i
    if start_line is not None and i > start_line and line.strip() and not line[0].isspace() and 'def ' in line:
        end_line = i
        break

if end_line is None:
    # Find the next method after calculate_consistency_score
    for i in range(start_line + 1, len(lines)):
        if lines[i].strip().startswith('def '):
            end_line = i
            break

# Create the improved method
improved_method = '''    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None
    ) -> float:
        """
        Calculate consistency based on rate of change.
        Improved to handle short time periods more reasonably.
        
        Args:
            weight: Current weight
            previous_weight: Previous weight
            time_diff_hours: Time since previous measurement
            
        Returns:
            Score from 0.0 (inconsistent) to 1.0 (consistent)
        """
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        
        # Use different thresholds based on time period to avoid
        # penalizing normal variations in short time windows
        if time_diff_hours < 6:
            # Within 6 hours: allow up to 3kg variation
            # (accounts for meals, hydration, bathroom visits)
            max_allowed = 3.0
            typical_allowed = 1.5
            
            if weight_diff <= typical_allowed:
                return 1.0
            elif weight_diff <= max_allowed:
                ratio = (weight_diff - typical_allowed) / (max_allowed - typical_allowed)
                return 1.0 - (0.3 * ratio)
            else:
                excess_ratio = (weight_diff - max_allowed) / max_allowed
                return 0.7 * np.exp(-2 * excess_ratio)
                
        elif time_diff_hours < 24:
            # Within a day: interpolate between short-term and daily limits
            hours_ratio = time_diff_hours / 24
            max_allowed = 3.0 + (PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG'] - 3.0) * hours_ratio
            typical_allowed = 1.5 + (PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG'] - 1.5) * hours_ratio
            
            if weight_diff <= typical_allowed:
                return 1.0
            elif weight_diff <= max_allowed:
                ratio = (weight_diff - typical_allowed) / (max_allowed - typical_allowed)
                return 1.0 - (0.4 * ratio)
            else:
                excess_ratio = (weight_diff - max_allowed) / max_allowed
                return 0.6 * np.exp(-2 * excess_ratio)
        
        else:
            # For longer periods: use daily rate as before
            daily_rate = (weight_diff / time_diff_hours) * 24
            max_daily = PHYSIOLOGICAL_LIMITS['MAX_DAILY_CHANGE_KG']
            typical_daily = PHYSIOLOGICAL_LIMITS['TYPICAL_DAILY_VARIATION_KG']
            
            if daily_rate <= typical_daily:
                return 1.0
            elif daily_rate <= max_daily:
                ratio = (daily_rate - typical_daily) / (max_daily - typical_daily)
                return 1.0 - (0.5 * ratio)
            else:
                excess_ratio = (daily_rate - max_daily) / max_daily
                score = 0.5 * np.exp(-2 * excess_ratio)
                return max(0.0, score)
    
'''

# Replace the method
new_lines = lines[:start_line] + [improved_method] + lines[end_line:]

# Write back
with open('src/quality_scorer.py', 'w') as f:
    f.writelines(new_lines)

print(f"Updated calculate_consistency_score method (lines {start_line+1} to {end_line})")
