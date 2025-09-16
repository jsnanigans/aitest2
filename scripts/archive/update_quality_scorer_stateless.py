"""
Update quality_scorer.py to be properly stateless
"""

# Read the current file
with open('src/quality_scorer.py', 'r') as f:
    content = f.read()

# Find and replace the calculate_consistency_score method
import re

# The improved stateless version
new_consistency_method = '''    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None
    ) -> float:
        """
        Calculate consistency based on rate of change.
        Improved to handle short time periods more reasonably.
        Stateless - uses only provided parameters.
        
        Args:
            weight: Current weight
            previous_weight: Previous weight from state
            time_diff_hours: Time since previous measurement
            
        Returns:
            Score from 0.0 (inconsistent) to 1.0 (consistent)
        """
        if previous_weight is None or time_diff_hours is None or time_diff_hours <= 0:
            return 0.8
        
        weight_diff = abs(weight - previous_weight)
        
        # Estimate baseline for percentage calculation
        baseline_weight = (weight + previous_weight) / 2
        weight_diff_percent = (weight_diff / baseline_weight) * 100
        
        # Time-aware thresholds based on research
        if time_diff_hours < 6:
            # Within 6 hours: allow normal variation
            typical_percent = 1.0
            max_percent = 2.5
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.3 * ratio)
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.7 * np.exp(-2 * excess)
                
        elif time_diff_hours < 24:
            # Within a day: interpolate thresholds
            hours_ratio = time_diff_hours / 24
            typical_percent = 1.0 + (2.0 - 1.0) * hours_ratio
            max_percent = 2.5 + (3.5 - 2.5) * hours_ratio
            
            if weight_diff_percent <= typical_percent:
                return 1.0
            elif weight_diff_percent <= max_percent:
                ratio = (weight_diff_percent - typical_percent) / (max_percent - typical_percent)
                return 1.0 - (0.4 * ratio)
            else:
                excess = (weight_diff_percent - max_percent) / max_percent
                return 0.6 * np.exp(-2 * excess)
        
        else:
            # For longer periods: use daily rate
            daily_rate_percent = weight_diff_percent / (time_diff_hours / 24)
            
            if daily_rate_percent <= 0.35:  # Normal weekly variation
                return 1.0
            elif daily_rate_percent <= 0.7:
                ratio = (daily_rate_percent - 0.35) / 0.35
                return 1.0 - (0.5 * ratio)
            else:
                excess = (daily_rate_percent - 0.7) / 0.7
                return 0.5 * np.exp(-2 * excess)
    '''

# Find the method and replace it
pattern = r'(    def calculate_consistency_score\([\s\S]*?)(?=\n    def )'
replacement = new_consistency_method

new_content = re.sub(pattern, new_consistency_method, content)

# Write back
with open('src/quality_scorer.py', 'w') as f:
    f.write(new_content)

print("Updated quality_scorer.py with stateless consistency scoring")
