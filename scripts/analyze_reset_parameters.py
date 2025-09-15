#!/usr/bin/env python3
"""
Comprehensive analysis script to:
1. Investigate why reset configs aren't being applied
2. Analyze user data to find optimal parameters
3. Test with real pipeline and provide debugging
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import toml
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import WeightProcessor
from src.database import ProcessorDatabase
from src.reset_manager import ResetManager, ResetType
from src.quality_scorer import QualityScorer
# from src.kalman_adaptive import AdaptiveKalmanFilter
# from src.utils import parse_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResetParameterAnalyzer:
    """Analyze and optimize reset parameters for different scenarios."""
    
    def __init__(self, csv_file: str, config_file: str = 'config.toml'):
        """Initialize analyzer with data and config."""
        self.csv_file = csv_file
        self.config_file = config_file
        self.config = toml.load(config_file)
        self.df = pd.read_csv(csv_file)
        self.df['effectiveDateTime'] = pd.to_datetime(self.df['effectiveDateTime'])
        self.df = self.df.sort_values(['user_id', 'effectiveDateTime'])
        
        # Analysis results
        self.user_analysis = {}
        self.parameter_performance = defaultdict(list)
        self.debug_logs = {}
        
    def analyze_user(self, user_id: str, verbose: bool = False) -> Dict[str, Any]:
        """Analyze a single user's data patterns."""
        user_data = self.df[self.df['user_id'] == user_id].copy()
        
        if len(user_data) < 5:
            return None
            
        analysis = {
            'user_id': user_id,
            'total_measurements': len(user_data),
            'date_range': (user_data['effectiveDateTime'].min(), user_data['effectiveDateTime'].max()),
            'sources': user_data['source_type'].value_counts().to_dict(),
            'gaps': [],
            'weight_changes': [],
            'initial_phase': {},
            'reset_scenarios': []
        }
        
        # Analyze gaps
        for i in range(1, len(user_data)):
            gap_days = (user_data.iloc[i]['effectiveDateTime'] - user_data.iloc[i-1]['effectiveDateTime']).days
            if gap_days > 1:
                analysis['gaps'].append({
                    'days': gap_days,
                    'before': user_data.iloc[i-1]['weight'],
                    'after': user_data.iloc[i]['weight'],
                    'weight_change': abs(user_data.iloc[i]['weight'] - user_data.iloc[i-1]['weight']),
                    'index': i
                })
        
        # Analyze weight changes
        weights = user_data['weight'].values
        for i in range(1, len(weights)):
            change = abs(weights[i] - weights[i-1])
            change_pct = change / weights[i-1] * 100
            analysis['weight_changes'].append({
                'absolute': change,
                'percent': change_pct,
                'source': user_data.iloc[i]['source_type']
            })
        
        # Analyze initial phase (first 30 measurements)
        initial_data = user_data.head(30)
        if len(initial_data) > 5:
            initial_weights = initial_data['weight'].values
            analysis['initial_phase'] = {
                'count': len(initial_data),
                'mean': np.mean(initial_weights),
                'std': np.std(initial_weights),
                'cv': np.std(initial_weights) / np.mean(initial_weights) * 100,
                'max_change': np.max(np.abs(np.diff(initial_weights))),
                'trend': np.polyfit(range(len(initial_weights)), initial_weights, 1)[0]
            }
        
        # Identify reset scenarios
        for gap in analysis['gaps']:
            if gap['days'] >= 30:
                analysis['reset_scenarios'].append({
                    'type': 'hard',
                    'gap_days': gap['days'],
                    'weight_change': gap['weight_change']
                })
        
        # Check for manual data scenarios
        manual_sources = ['internal-questionnaire', 'patient-upload', 'care-team-upload']
        for i, row in user_data.iterrows():
            if row['source_type'] in manual_sources:
                idx = user_data.index.get_loc(i)
                if idx > 0:
                    prev_weight = user_data.iloc[idx-1]['weight']
                    weight_change = abs(row['weight'] - prev_weight)
                    if weight_change >= 5:
                        analysis['reset_scenarios'].append({
                            'type': 'soft',
                            'weight_change': weight_change,
                            'source': row['source_type']
                        })
        
        return analysis
    
    def test_parameters(self, user_id: str, params: Dict[str, Any], 
                       reset_type: ResetType = ResetType.INITIAL) -> Dict[str, Any]:
        """Test specific parameters on a user's data."""
        user_data = self.df[self.df['user_id'] == user_id]
        
        if len(user_data) < 5:
            return None
        
        # Create temporary config with test parameters
        test_config = self.config.copy()
        reset_key = reset_type.value
        if 'kalman' not in test_config:
            test_config['kalman'] = {}
        if 'reset' not in test_config['kalman']:
            test_config['kalman']['reset'] = {}
        test_config['kalman']['reset'][reset_key] = params
        
        # Initialize processor and database
        db = ProcessorDatabase()
        processor = WeightProcessor()
        
        # Process measurements
        accepted = 0
        rejected = 0
        rejection_reasons = []
        
        for _, row in user_data.iterrows():
            state = db.get_user_state(user_id)
            
            result = processor.process_measurement(
                user_id=user_id,
                weight=row['weight'],
                timestamp=row['effectiveDateTime'],
                source=row['source_type'],
                state=state,
                config=test_config
            )
            
            if result['accepted']:
                accepted += 1
                db.update_user_state(user_id, result['state'])
            else:
                rejected += 1
                rejection_reasons.append(result.get('rejection_reason', 'unknown'))
        
        return {
            'user_id': user_id,
            'parameters': params,
            'accepted': accepted,
            'rejected': rejected,
            'acceptance_rate': accepted / (accepted + rejected) * 100,
            'rejection_reasons': rejection_reasons
        }
    
    def optimize_parameters(self, reset_type: ResetType = ResetType.INITIAL,
                          sample_users: int = 10) -> Dict[str, Any]:
        """Find optimal parameters for a reset type."""
        
        # Get sample of users with relevant scenarios
        relevant_users = []
        for user_id in self.df['user_id'].unique()[:sample_users * 2]:
            analysis = self.analyze_user(user_id)
            if analysis and len(analysis['reset_scenarios']) > 0:
                relevant_users.append(user_id)
                if len(relevant_users) >= sample_users:
                    break
        
        if not relevant_users:
            logger.warning(f"No users found with {reset_type.value} reset scenarios")
            return None
        
        # Parameter ranges to test
        param_ranges = {
            'initial_variance_multiplier': [5, 10, 20, 50],
            'weight_noise_multiplier': [10, 30, 50, 100],
            'trend_noise_multiplier': [100, 300, 500, 1000],
            'observation_noise_multiplier': [0.1, 0.3, 0.5, 1.0, 2.0],
            'adaptation_measurements': [10, 20, 30, 50],
            'adaptation_days': [7, 14, 21, 30, 60, 90],
            'adaptation_decay_rate': [1.0, 2.0, 3.0, 5.0],
            'quality_acceptance_threshold': [0.0, 0.1, 0.25, 0.35, 0.45]
        }
        
        best_params = None
        best_score = 0
        
        # Grid search (simplified - in practice would use optimization)
        for ivm in param_ranges['initial_variance_multiplier']:
            for wnm in param_ranges['weight_noise_multiplier']:
                for am in param_ranges['adaptation_measurements']:
                    for qat in param_ranges['quality_acceptance_threshold']:
                        params = {
                            'initial_variance_multiplier': ivm,
                            'weight_noise_multiplier': wnm,
                            'trend_noise_multiplier': 500,  # Fixed for speed
                            'observation_noise_multiplier': 0.3,
                            'adaptation_measurements': am,
                            'adaptation_days': am * 2,  # Proportional
                            'adaptation_decay_rate': 3.0,
                            'quality_acceptance_threshold': qat,
                            'quality_safety_weight': 0.35,
                            'quality_plausibility_weight': 0.25,
                            'quality_consistency_weight': 0.25,
                            'quality_reliability_weight': 0.15
                        }
                        
                        # Test on sample users
                        total_score = 0
                        for user_id in relevant_users[:3]:  # Quick test
                            result = self.test_parameters(user_id, params, reset_type)
                            if result:
                                # Score based on acceptance rate with penalty for too high
                                if result['acceptance_rate'] > 95:
                                    score = 95 - (result['acceptance_rate'] - 95) * 2
                                else:
                                    score = result['acceptance_rate']
                                total_score += score
                        
                        avg_score = total_score / 3
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
                            logger.info(f"New best score: {best_score:.2f}% with params: {params}")
        
        return {
            'reset_type': reset_type.value,
            'best_parameters': best_params,
            'best_score': best_score,
            'tested_users': relevant_users[:3]
        }
    
    def debug_config_loading(self) -> Dict[str, Any]:
        """Debug why config values aren't being applied."""
        debug_info = {
            'config_file': self.config_file,
            'config_loaded': False,
            'reset_config': {},
            'issues': []
        }
        
        try:
            # Load config
            config = toml.load(self.config_file)
            debug_info['config_loaded'] = True
            
            # Check reset config structure
            if 'kalman' in config:
                if 'reset' in config['kalman']:
                    debug_info['reset_config'] = config['kalman']['reset']
                    
                    # Check each reset type
                    for reset_type in ['initial', 'hard', 'soft']:
                        if reset_type in config['kalman']['reset']:
                            rt_config = config['kalman']['reset'][reset_type]
                            debug_info[f'{reset_type}_config'] = rt_config
                            
                            # Verify key parameters
                            expected_keys = [
                                'initial_variance_multiplier',
                                'weight_noise_multiplier', 
                                'adaptation_measurements',
                                'quality_acceptance_threshold'
                            ]
                            
                            for key in expected_keys:
                                if key not in rt_config:
                                    debug_info['issues'].append(
                                        f"Missing {key} in {reset_type} config"
                                    )
                        else:
                            debug_info['issues'].append(f"Missing {reset_type} reset config")
                else:
                    debug_info['issues'].append("Missing kalman.reset section")
            else:
                debug_info['issues'].append("Missing kalman section")
            
            # Test parameter loading with ResetManager
            for reset_type in [ResetType.INITIAL, ResetType.HARD, ResetType.SOFT]:
                params = ResetManager.get_reset_parameters(reset_type, config)
                debug_info[f'{reset_type.value}_loaded_params'] = params
                
                # Check if config values override defaults
                config_values = config.get('kalman', {}).get('reset', {}).get(reset_type.value, {})
                for key, value in config_values.items():
                    if key in params and params[key] != value:
                        debug_info['issues'].append(
                            f"{reset_type.value}.{key}: config={value}, loaded={params[key]}"
                        )
            
        except Exception as e:
            debug_info['error'] = str(e)
            debug_info['issues'].append(f"Error loading config: {e}")
        
        return debug_info
    
    def generate_report(self, output_dir: str = 'output/analysis'):
        """Generate comprehensive analysis report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_file': self.csv_file,
            'config_file': self.config_file,
            'config_debug': self.debug_config_loading(),
            'user_analyses': {},
            'parameter_optimization': {},
            'recommendations': {}
        }
        
        # Analyze sample of users
        sample_users = list(self.df['user_id'].unique()[:20])
        for user_id in sample_users:
            analysis = self.analyze_user(user_id)
            if analysis:
                report['user_analyses'][user_id] = analysis
        
        # Optimize parameters for each reset type
        for reset_type in [ResetType.INITIAL, ResetType.HARD, ResetType.SOFT]:
            logger.info(f"Optimizing {reset_type.value} parameters...")
            optimization = self.optimize_parameters(reset_type, sample_users=5)
            if optimization:
                report['parameter_optimization'][reset_type.value] = optimization
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(report)
        
        # Save report
        report_file = Path(output_dir) / f'reset_analysis_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_file}")
        return report
    
    def generate_recommendations(self, report: Dict) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        recommendations = {
            'initial': {},
            'hard': {},
            'soft': {}
        }
        
        # Analyze user patterns
        initial_cvs = []
        gap_patterns = []
        
        for user_id, analysis in report['user_analyses'].items():
            if analysis and 'initial_phase' in analysis and analysis['initial_phase']:
                initial_cvs.append(analysis['initial_phase'].get('cv', 0))
            if analysis and 'gaps' in analysis:
                for gap in analysis['gaps']:
                    gap_patterns.append(gap)
        
        # Initial reset recommendations
        if initial_cvs:
            avg_cv = np.mean(initial_cvs)
            if avg_cv > 10:  # High variability
                recommendations['initial'] = {
                    'initial_variance_multiplier': 20,
                    'weight_noise_multiplier': 100,
                    'adaptation_measurements': 50,
                    'quality_acceptance_threshold': 0.0,
                    'reasoning': f"High initial variability (CV={avg_cv:.1f}%), need very adaptive parameters"
                }
            else:
                recommendations['initial'] = {
                    'initial_variance_multiplier': 10,
                    'weight_noise_multiplier': 50,
                    'adaptation_measurements': 30,
                    'quality_acceptance_threshold': 0.1,
                    'reasoning': f"Moderate initial variability (CV={avg_cv:.1f}%)"
                }
        
        # Hard reset recommendations
        if gap_patterns:
            avg_gap_change = np.mean([g['weight_change'] for g in gap_patterns])
            if avg_gap_change > 10:
                recommendations['hard'] = {
                    'initial_variance_multiplier': 10,
                    'weight_noise_multiplier': 30,
                    'adaptation_measurements': 15,
                    'quality_acceptance_threshold': 0.25,
                    'reasoning': f"Large weight changes after gaps (avg={avg_gap_change:.1f}kg)"
                }
            else:
                recommendations['hard'] = {
                    'initial_variance_multiplier': 5,
                    'weight_noise_multiplier': 20,
                    'adaptation_measurements': 10,
                    'quality_acceptance_threshold': 0.35,
                    'reasoning': f"Moderate weight changes after gaps (avg={avg_gap_change:.1f}kg)"
                }
        
        # Soft reset recommendations (always gentle)
        recommendations['soft'] = {
            'initial_variance_multiplier': 2,
            'weight_noise_multiplier': 5,
            'adaptation_measurements': 15,
            'quality_acceptance_threshold': 0.45,
            'reasoning': "Manual data entry should be trusted with gentle adaptation"
        }
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Analyze and optimize reset parameters')
    parser.add_argument('--csv', default='./data/2025-09-05_nocon.csv',
                       help='Input CSV file')
    parser.add_argument('--config', default='config.toml',
                       help='Configuration file')
    parser.add_argument('--output', default='output/analysis',
                       help='Output directory for reports')
    parser.add_argument('--user', help='Analyze specific user')
    parser.add_argument('--optimize', choices=['initial', 'hard', 'soft'],
                       help='Optimize specific reset type')
    parser.add_argument('--debug-config', action='store_true',
                       help='Debug configuration loading')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = ResetParameterAnalyzer(args.csv, args.config)
    
    if args.debug_config:
        # Debug configuration loading
        debug_info = analyzer.debug_config_loading()
        print("\n=== Configuration Debug Info ===")
        print(json.dumps(debug_info, indent=2))
        
        if debug_info['issues']:
            print("\n=== Issues Found ===")
            for issue in debug_info['issues']:
                print(f"  - {issue}")
    
    elif args.user:
        # Analyze specific user
        analysis = analyzer.analyze_user(args.user, verbose=args.verbose)
        if analysis:
            print(f"\n=== Analysis for User {args.user} ===")
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(f"No data or insufficient data for user {args.user}")
    
    elif args.optimize:
        # Optimize specific reset type
        reset_type = ResetType[args.optimize.upper()]
        result = analyzer.optimize_parameters(reset_type)
        if result:
            print(f"\n=== Optimization Results for {args.optimize} ===")
            print(json.dumps(result, indent=2, default=str))
    
    else:
        # Generate full report
        print("Generating comprehensive analysis report...")
        report = analyzer.generate_report(args.output)
        
        # Print summary
        print("\n=== Analysis Summary ===")
        print(f"Analyzed {len(report['user_analyses'])} users")
        print(f"Config issues found: {len(report['config_debug'].get('issues', []))}")
        
        if report['config_debug'].get('issues'):
            print("\nConfiguration Issues:")
            for issue in report['config_debug']['issues']:
                print(f"  - {issue}")
        
        print("\n=== Recommended Parameters ===")
        for reset_type, rec in report['recommendations'].items():
            if rec:
                print(f"\n{reset_type.upper()} Reset:")
                print(f"  Reasoning: {rec.get('reasoning', 'N/A')}")
                for key, value in rec.items():
                    if key != 'reasoning':
                        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
