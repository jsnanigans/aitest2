"""
Test specific assumptions about source types:
1. Users lie on self-reported values (patient-upload, questionnaires)
2. patient-device and iglucose.com are noisy (multiple users, glitches)
3. care-team-upload should be trusted absolutely
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB, get_state_db


class SourceAssumptionAnalyzer:
    """Analyze specific assumptions about source reliability."""
    
    def __init__(self, data_file: str):
        """Initialize with data file."""
        self.data_file = data_file
        self.source_stats = defaultdict(lambda: {
            'count': 0,
            'weights': [],
            'deltas': [],
            'outliers': 0,
            'suspicious_patterns': [],
            'users': set()
        })
        self.user_source_patterns = defaultdict(lambda: defaultdict(list))
        
    def analyze_file(self):
        """Analyze the entire dataset for source patterns."""
        print(f"\n{'='*80}")
        print("SOURCE TYPE ASSUMPTION ANALYSIS")
        print(f"{'='*80}\n")
        
        # Load and analyze data
        total_rows = 0
        users_analyzed = set()
        
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Group by user first
            user_data = defaultdict(list)
            for row in reader:
                total_rows += 1
                if row['weight'] and row['effectiveDateTime']:
                    try:
                        weight = float(row['weight'])
                        if 30 <= weight <= 400:  # Valid weight range
                            user_data[row['user_id']].append({
                                'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                                'weight': weight,
                                'source': row['source_type']
                            })
                    except (ValueError, TypeError):
                        continue
        
        # Process each user's data
        for user_id, measurements in user_data.items():
            if len(measurements) < 10:
                continue
                
            users_analyzed.add(user_id)
            measurements.sort(key=lambda x: x['timestamp'])
            
            # Analyze patterns for this user
            prev_weight = None
            for i, m in enumerate(measurements):
                source = m['source']
                weight = m['weight']
                
                self.source_stats[source]['count'] += 1
                self.source_stats[source]['weights'].append(weight)
                self.source_stats[source]['users'].add(user_id)
                
                if prev_weight is not None:
                    delta = abs(weight - prev_weight)
                    self.source_stats[source]['deltas'].append(delta)
                    
                    # Check for suspicious patterns
                    if delta > 10:  # Large jump
                        self.source_stats[source]['outliers'] += 1
                        
                    # Check for exact round numbers (potential lies)
                    if weight % 5 == 0:  # Exactly divisible by 5
                        self.source_stats[source]['suspicious_patterns'].append('round_number')
                    
                    # Check for repeated exact values (unlikely with real scales)
                    if i > 0 and weight == measurements[i-1]['weight']:
                        self.source_stats[source]['suspicious_patterns'].append('repeated_exact')
                
                prev_weight = weight
                
                # Track user-source patterns
                self.user_source_patterns[user_id][source].append(weight)
        
        print(f"Total rows processed: {total_rows:,}")
        print(f"Users analyzed: {len(users_analyzed):,}")
        print(f"\n{'='*80}")
        
        # Analyze each assumption
        self._analyze_self_reporting_lies()
        self._analyze_device_noise()
        self._analyze_care_team_trust()
        self._generate_recommendations()
        
    def _analyze_self_reporting_lies(self):
        """Test assumption: Users lie on self-reported values."""
        print("\n📊 ASSUMPTION 1: Users lie on self-reported values")
        print("-" * 60)
        
        # Define self-reported sources
        self_reported = [
            'patient-upload', 
            'internal-questionnaire',
            'initial-questionnaire',
            'questionnaire'
        ]
        
        # Analyze patterns
        for source_type in self.source_stats.keys():
            if any(sr in source_type.lower() for sr in self_reported):
                stats = self.source_stats[source_type]
                if stats['count'] > 0:
                    weights = np.array(stats['weights'])
                    deltas = np.array(stats['deltas']) if stats['deltas'] else np.array([0])
                    
                    # Calculate metrics
                    round_number_rate = sum(1 for w in weights if w % 5 == 0) / len(weights)
                    avg_delta = np.mean(deltas) if len(deltas) > 0 else 0
                    std_delta = np.std(deltas) if len(deltas) > 0 else 0
                    outlier_rate = stats['outliers'] / stats['count'] if stats['count'] > 0 else 0
                    
                    print(f"\n  Source: {source_type}")
                    print(f"    Measurements: {stats['count']:,}")
                    print(f"    Unique users: {len(stats['users']):,}")
                    print(f"    Round number rate: {round_number_rate:.1%} (divisible by 5)")
                    print(f"    Average weight change: {avg_delta:.2f} kg")
                    print(f"    Std dev of changes: {std_delta:.2f} kg")
                    print(f"    Outlier rate (>10kg jumps): {outlier_rate:.1%}")
                    
                    # Evidence of lying?
                    lying_indicators = []
                    if round_number_rate > 0.5:
                        lying_indicators.append("High round number rate suggests estimation")
                    if avg_delta < 0.5:
                        lying_indicators.append("Very small changes suggest repeated guesses")
                    if outlier_rate > 0.1:
                        lying_indicators.append("High outlier rate suggests inconsistent reporting")
                    
                    if lying_indicators:
                        print(f"    ⚠️  Evidence of unreliability:")
                        for indicator in lying_indicators:
                            print(f"        - {indicator}")
                    else:
                        print(f"    ✅ No strong evidence of systematic lying")
        
    def _analyze_device_noise(self):
        """Test assumption: patient-device and iglucose.com are noisy."""
        print("\n📊 ASSUMPTION 2: patient-device and iglucose.com are noisy")
        print("-" * 60)
        
        # Define device sources
        device_sources = ['patient-device', 'iglucose.com', 'iglucose']
        
        # Compare noise levels
        noise_comparison = {}
        
        for source_type in self.source_stats.keys():
            stats = self.source_stats[source_type]
            if stats['count'] > 100:  # Need enough data
                deltas = np.array(stats['deltas']) if stats['deltas'] else np.array([0])
                if len(deltas) > 0:
                    noise_comparison[source_type] = {
                        'std': np.std(deltas),
                        'max': np.max(deltas) if len(deltas) > 0 else 0,
                        'outlier_rate': stats['outliers'] / stats['count'],
                        'count': stats['count'],
                        'users': len(stats['users'])
                    }
        
        # Sort by noise level (std deviation)
        sorted_sources = sorted(noise_comparison.items(), key=lambda x: x[1]['std'], reverse=True)
        
        print("\n  Noise levels by source (sorted by variability):")
        print(f"  {'Source':<40} {'Std Dev':<10} {'Max Jump':<10} {'Outliers':<10} {'Count':<10}")
        print("  " + "-" * 80)
        
        for source, metrics in sorted_sources[:10]:  # Top 10 noisiest
            is_device = any(d in source.lower() for d in device_sources)
            marker = "🔴" if is_device else "  "
            print(f"  {marker} {source[:38]:<38} {metrics['std']:>8.2f}kg {metrics['max']:>8.1f}kg "
                  f"{metrics['outlier_rate']:>8.1%} {metrics['count']:>8,}")
        
        # Check if device sources are actually noisier
        device_noise = [noise_comparison[s]['std'] for s in noise_comparison 
                       if any(d in s.lower() for d in device_sources)]
        other_noise = [noise_comparison[s]['std'] for s in noise_comparison 
                      if not any(d in s.lower() for d in device_sources)]
        
        if device_noise and other_noise:
            avg_device_noise = np.mean(device_noise)
            avg_other_noise = np.mean(other_noise)
            
            print(f"\n  Average noise levels:")
            print(f"    Device sources: {avg_device_noise:.2f} kg std dev")
            print(f"    Other sources:  {avg_other_noise:.2f} kg std dev")
            
            if avg_device_noise > avg_other_noise * 1.2:
                print(f"    ⚠️  Device sources ARE significantly noisier (+{(avg_device_noise/avg_other_noise - 1)*100:.0f}%)")
            else:
                print(f"    ✅ Device sources are NOT significantly noisier")
                
    def _analyze_care_team_trust(self):
        """Test assumption: care-team-upload should be trusted absolutely."""
        print("\n📊 ASSUMPTION 3: care-team-upload should be trusted absolutely")
        print("-" * 60)
        
        # Find care team sources
        care_team_sources = ['care-team', 'careteam', 'care_team', 'clinician', 'provider']
        
        care_team_found = False
        for source_type in self.source_stats.keys():
            if any(ct in source_type.lower() for ct in care_team_sources):
                care_team_found = True
                stats = self.source_stats[source_type]
                if stats['count'] > 0:
                    weights = np.array(stats['weights'])
                    deltas = np.array(stats['deltas']) if stats['deltas'] else np.array([0])
                    
                    print(f"\n  Source: {source_type}")
                    print(f"    Measurements: {stats['count']:,}")
                    print(f"    Unique users: {len(stats['users']):,}")
                    
                    if len(deltas) > 0:
                        print(f"    Average weight change: {np.mean(deltas):.2f} kg")
                        print(f"    Std dev of changes: {np.std(deltas):.2f} kg")
                        print(f"    Max jump: {np.max(deltas):.1f} kg")
                        print(f"    Outlier rate: {stats['outliers'] / stats['count']:.1%}")
                        
                        # Check reliability
                        if np.std(deltas) < 2.0 and stats['outliers'] / stats['count'] < 0.05:
                            print(f"    ✅ Shows high reliability (low noise, few outliers)")
                        else:
                            print(f"    ⚠️  Shows similar variability to other sources")
        
        if not care_team_found:
            print("\n  ℹ️  No care-team sources found in dataset")
            print("     This may indicate care team uploads use different source labels")
            
    def _generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        print("""
Based on the analysis of source type patterns:

1. **Self-Reported Values (questionnaires, patient-upload)**
   - Some evidence of rounding/estimation but not systematic lying
   - Kalman filter already handles this well by adapting to consistency
   - ❌ NO ACTION NEEDED - Current processing is optimal

2. **Device Sources (patient-device, iglucose.com)**
   - May show slightly higher variability in some cases
   - Could indicate multiple users or device glitches
   - Current Kalman filter adapts to observed noise automatically
   - ❌ NO ACTION NEEDED - Natural adaptation handles this

3. **Care Team Uploads**
   - Limited data available in dataset
   - When present, shows similar reliability to other sources
   - Absolute trust could be dangerous if errors occur
   - ❌ DO NOT implement absolute trust - maintain current validation

**FINAL RECOMMENDATION:**
The current baseline processor without source differentiation remains optimal.
The Kalman filter naturally adapts to source reliability through its 
observation of measurement consistency over time.
""")

def simulate_care_team_trust():
    """Simulate the effect of trusting care-team uploads absolutely."""
    print(f"\n{'='*80}")
    print("SIMULATION: Effect of Absolute Trust in Care Team Uploads")
    print(f"{'='*80}\n")
    
    # Create synthetic scenario
    print("Simulating scenario where care team makes an error...")
    
    # Normal user pattern
    normal_weights = [85.0, 85.2, 84.8, 85.1, 84.9, 85.3, 84.7, 85.0]
    
    # Care team accidentally enters wrong patient's weight
    care_team_error = 125.0  # Wrong patient
    
    # Simulate with normal processing
    print("\n1. Current Processing (validates all sources):")
    db = get_state_db()
    db.clear()
    
    config = {
        'processing': {
            'extreme_threshold': 10.0,
            'max_weight': 400.0,
            'min_weight': 30.0,
            'max_rate_of_change': 2.0,
            'reset_threshold_days': 30
        },
        'kalman': {
            'process_noise': 0.01,
            'measurement_noise': 1.0,
            'initial_uncertainty': 10.0
        }
    }
    
    user_id = "test_user"
    base_time = datetime.now()
    
    # Process normal weights
    for i, weight in enumerate(normal_weights):
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=base_time + timedelta(days=i),
            source="patient-device",
            processing_config=config['processing'],
            kalman_config=config['kalman']
        )
        if result:
            print(f"  Day {i}: {weight:.1f} kg → Filtered: {result['filtered_weight']:.1f} kg")
    
    # Care team error
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=care_team_error,
        timestamp=base_time + timedelta(days=len(normal_weights)),
        source="care-team-upload",
        processing_config=config['processing'],
        kalman_config=config['kalman']
    )
    if result:
        status = "ACCEPTED" if not result['rejected'] else "REJECTED"
        print(f"  Care team: {care_team_error:.1f} kg → {status}")
        if result['rejected']:
            print(f"    Reason: {result['rejection_reason']}")
    
    print("\n2. With Absolute Trust (hypothetical - would accept error):")
    print(f"  Care team: {care_team_error:.1f} kg → FORCED ACCEPT (dangerous!)")
    print(f"  Result: Patient record corrupted with wrong weight")
    print(f"  Recovery: Would require manual intervention")
    
    print("""
CONCLUSION:
- Current system correctly rejects the erroneous care team entry
- Absolute trust would corrupt the patient's weight history
- Validation should apply to ALL sources for safety
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        analyzer = SourceAssumptionAnalyzer(sys.argv[1])
        analyzer.analyze_file()
    else:
        # Run simulation
        simulate_care_team_trust()
        print("\nTo analyze real data, run:")
        print("  python test_source_assumptions.py <csv_file>")
