"""Fix consistency scoring to match test expectations while keeping improvements"""

with open('src/quality_scorer.py', 'r') as f:
    content = f.read()

# Replace the consistency score method with a balanced version
new_method = '''    def calculate_consistency_score(
        self,
        weight: float,
        previous_weight: Optional[float] = None,
        time_diff_hours: Optional[float] = None
    ) -> float:
        """
        Calculate consistency based on rate of change.
        Improved to handle short time periods based on research.
        STATELESS - Only uses previous weight and time.
        
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
        baseline_weight = (weight + previous_weight) / 2
        weight_diff_percent = (weight_diff / baseline_weight) * 100
        
        # Time-aware thresholds
        if time_diff_hours < 6:
            # Within 6 hours: very lenient for normal fluctuations
            if weight_diff <= 3.0:  # Up to 3kg is normal within hours
                return 1.0
            else:
                excess = (weight_diff - 3.0) / 3.0
                return max(0.0, 0.7 * np.exp(-2 * excess))
                
        elif time_diff_hours < 24:
            # Within a day: use hybrid approach
            # Small absolute changes always OK
            if weight_diff <= 2.0:
                return 1.0
            # Moderate changes get high scores
            elif weight_diff <= 4.0:
                # Linear decay from 1.0 to 0.8
                return 1.0 - 0.1 * (weight_diff - 2.0)
            # Large changes use percentage
            elif weight_diff_percent <= 5.0:
                return 0.7
            else:
                excess = (weight_diff_percent - 5.0) / 5.0
                return max(0.0, 0.5 * np.exp(-2 * excess))
        
        else:
            # For longer periods: use daily rate
            daily_rate = weight_diff / (time_diff_hours / 24)
            
            # Research-based but practical thresholds
            if daily_rate <= 2.0:  # Up to 2kg/day for longer periods
                return 1.0
            elif daily_rate <= 4.0:
                # Linear decay from 1.0 to 0.5
                return 1.0 - 0.25 * (daily_rate - 2.0)
            elif daily_rate <= 6.44:  # Physiological max
                return 0.5 - 0.3 * ((daily_rate - 4.0) / 2.44)
            else:
                excess = (daily_rate - 6.44) / 6.44
                return max(0.0, 0.2 * np.exp(-2 * excess))
    '''

# Find and replace the method
import re
pattern = r'def calculate_consistency_score\([^}]+?\n(?:.*?\n)*?(?=    def |\Z)'
content = re.sub(pattern, new_method + '\n', content, count=1)

with open('src/quality_scorer.py', 'w') as f:
    f.write(content)

print("Fixed consistency scoring to balance test compatibility with improvements")
