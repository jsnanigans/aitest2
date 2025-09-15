"""
Fix consistency scoring to balance percentage and absolute changes
"""

# Read the file
with open('src/quality_scorer.py', 'r') as f:
    lines = f.readlines()

# Find the consistency method
start = None
for i, line in enumerate(lines):
    if 'def calculate_consistency_score(' in line:
        start = i
        break

# Find the end of the method
end = None
if start is not None:
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i][0].isspace() and 'def ' in lines[i]:
            end = i
            break

if end is None:
    end = len(lines)

# New balanced consistency method
new_method = '''    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None,
        recent_weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate consistency based on rate of change.
        Balances percentage-based and absolute thresholds.
        STATELESS - Uses parameters for all context.
        
        Args:
            weight: Current weight
            previous_weight: Previous weight from state
            time_diff_hours: Time since previous measurement
            recent_weights: Recent weights for baseline estimation
            
        Returns:
            Score from 0.0 (inconsistent) to 1.0 (consistent)
        """
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        
        # For backward compatibility with tests, use hybrid approach
        # Small absolute changes are always acceptable
        if weight_diff <= 0.5 and time_diff_hours >= 12:
            return 1.0
        elif weight_diff <= 1.0 and time_diff_hours >= 24:
            return 1.0
        
        # Estimate baseline for percentage calculation (stateless)
        if recent_weights and len(recent_weights) >= 3:
            baseline_weight = np.median(recent_weights[-10:])
        else:
            baseline_weight = (weight + previous_weight) / 2
        
        weight_diff_percent = (weight_diff / baseline_weight) * 100
        
        # Time-aware thresholds based on research
        if time_diff_hours < 6:
            # Within 6 hours: allow normal variation
            if weight_diff <= 2.0:  # Absolute cap
                typical_percent = 1.5
                max_percent = 3.0
            else:
                typical_percent = 1.0
                max_percent = 2.0
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.3 * ratio)
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.7 * np.exp(-2 * excess)
                
        elif time_diff_hours < 24:
            # Within a day: use both absolute and percentage
            if weight_diff <= 3.0:  # Reasonable daily change
                return max(0.8, 1.0 - (weight_diff / 15.0))
            
            # For larger changes, use percentage
            if weight_diff_percent <= 2.0:
                return 1.0
            elif weight_diff_percent <= 4.0:
                ratio = (weight_diff_percent - 2.0) / 2.0
                return 1.0 - (0.4 * ratio)
            else:
                excess = (weight_diff_percent - 4.0) / 4.0
                return 0.6 * np.exp(-2 * excess)
        
        else:
            # For longer periods: use daily rate
            daily_rate = weight_diff / (time_diff_hours / 24)
            daily_rate_percent = weight_diff_percent / (time_diff_hours / 24)
            
            # Use the more lenient of the two approaches
            if daily_rate <= 2.0:  # Absolute daily rate
                abs_score = 1.0
            elif daily_rate <= 4.0:
                abs_score = 1.0 - ((daily_rate - 2.0) / 4.0)
            else:
                abs_score = 0.5 * np.exp(-0.5 * (daily_rate - 4.0))
            
            if daily_rate_percent <= 0.5:  # Percentage daily rate
                pct_score = 1.0
            elif daily_rate_percent <= 1.0:
                pct_score = 1.0 - ((daily_rate_percent - 0.5) / 1.0)
            else:
                pct_score = 0.5 * np.exp(-0.5 * (daily_rate_percent - 1.0))
            
            return max(abs_score, pct_score)
    
'''

# Replace the method
if start is not None:
    new_lines = lines[:start] + [new_method] + lines[end:]
    
    with open('src/quality_scorer.py', 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated consistency scoring to balance percentage and absolute thresholds")
else:
    print("Could not find method to update")
