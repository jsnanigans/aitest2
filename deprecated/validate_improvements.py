#!/usr/bin/env python3
"""
Validate the implemented fixes by processing the challenging users with enhanced validation.
"""

import json
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict

sys.path.append('.')
from src.filters.enhanced_validation_gate import EnhancedValidationGate
from src.filters.multimodal_detector import MultimodalDetector

class ImprovedProcessor:
    def __init__(self):
        self.validation_gate = EnhancedValidationGate({
            'max_deviation_pct': 0.20,  # Allow 20% deviation
            'same_day_threshold_kg': 10.0,  # Allow 10kg difference on same day
            'rapid_change_threshold_pct': 0.20,  # Allow 20% rapid change
            'rapid_change_hours': 24,
            'outlier_z_score': 3.0,  # Less strict outlier detection
            'min_readings_for_stats': 10,  # Need more data before statistical checks
            'future_date_tolerance_days': 1,
            'duplicate_threshold_kg': 0.5
        })
        
        self.multimodal_detector = MultimodalDetector({
            'min_readings_for_detection': 30,
            'cluster_gap_kg': 15.0,
            'min_cluster_size': 5,
            'max_components': 3,
            'enable_auto_split': True
        })
        
        self.test_users = {
            '0040872d-333a-4ace-8c5a-b2fcd056e65a': 'madness_case',
            '0675ed39-53be-480b-baa6-fa53fc33709f': 'multi_person',
            '055b0c48-d5b4-44cb-8772-48154999e6c3': 'outliers_same'
        }
        
        self.results = {}
    
    def process_user(self, user_id: str, label: str):
        """Process a single user with all fixes applied"""
        
        raw_readings = []
        validated_readings = []
        rejected_readings = []
        deduplicated = []
        
        with open('./2025-09-05_optimized.csv', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5 and parts[0] == user_id:
                    reading = {
                        'date': datetime.fromisoformat(parts[1].replace(' ', 'T')),
                        'source': parts[2],
                        'weight': float(parts[3]),
                        'unit': parts[4]
                    }
                    raw_readings.append(reading)
        
        raw_readings.sort(key=lambda x: x['date'])
        
        for reading in raw_readings:
            
            if self.validation_gate.should_deduplicate(user_id, reading):
                deduplicated.append(reading)
                continue
            
            is_valid, reason = self.validation_gate.validate_reading(user_id, reading)
            
            if is_valid:
                validated_readings.append(reading)
            else:
                rejected_readings.append({'reading': reading, 'reason': reason})
        
        weights = [r['weight'] for r in validated_readings]
        if len(weights) >= 30:
            multimodal_result = self.multimodal_detector.detect_multimodal(user_id, weights)
        else:
            multimodal_result = {'is_multimodal': False, 'num_clusters': 1, 'reason': 'Insufficient validated data'}
        
        validation_stats = self.validation_gate.get_stats(user_id)
        
        result = {
            'user_id': user_id,
            'label': label,
            'raw_count': len(raw_readings),
            'validated_count': len(validated_readings),
            'rejected_count': len(rejected_readings),
            'deduplicated_count': len(deduplicated),
            'reduction_pct': (1 - len(validated_readings) / len(raw_readings)) * 100 if raw_readings else 0,
            'validation_stats': validation_stats,
            'multimodal_detection': multimodal_result,
            'sample_rejections': rejected_readings[:5],
            'final_stats': self._calculate_final_stats(validated_readings)
        }
        
        if multimodal_result.get('is_multimodal'):
            virtual_users = defaultdict(list)
            for reading in validated_readings:
                virtual_id = self.multimodal_detector.get_virtual_user_id(user_id, reading['weight'])
                virtual_users[virtual_id].append(reading)
            
            result['virtual_users'] = {}
            for vid, vreadings in virtual_users.items():
                vweights = [r['weight'] for r in vreadings]
                result['virtual_users'][vid] = {
                    'count': len(vreadings),
                    'mean': np.mean(vweights),
                    'std': np.std(vweights),
                    'range': (min(vweights), max(vweights))
                }
        
        return result
    
    def _calculate_final_stats(self, readings):
        """Calculate statistics on cleaned data"""
        if not readings:
            return {}
        
        weights = [r['weight'] for r in readings]
        dates = [r['date'] for r in readings]
        
        rapid_changes = 0
        for i in range(1, len(readings)):
            time_diff = (dates[i] - dates[i-1]).total_seconds() / 3600
            weight_diff = abs(weights[i] - weights[i-1])
            pct_change = weight_diff / weights[i-1] * 100 if weights[i-1] > 0 else 0
            if time_diff <= 24 and pct_change > 15:
                rapid_changes += 1
        
        return {
            'mean_weight': np.mean(weights),
            'std_deviation': np.std(weights),
            'weight_range': max(weights) - min(weights),
            'date_range': f"{dates[0].date()} to {dates[-1].date()}",
            'rapid_changes_remaining': rapid_changes,
            'cv': np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
        }
    
    def run_validation(self):
        """Run validation on all test users"""
        print("="*80)
        print("DATA QUALITY VALIDATION - AFTER FIXES")
        print("="*80)
        
        for user_id, label in self.test_users.items():
            print(f"\n{'-'*80}")
            print(f"USER: {user_id}")
            print(f"LABEL: {label.upper()}")
            print(f"{'-'*80}")
            
            result = self.process_user(user_id, label)
            self.results[label] = result
            
            print(f"\nBEFORE FIXES:")
            print(f"  Raw readings: {result['raw_count']}")
            
            print(f"\nAFTER FIXES:")
            print(f"  Validated readings: {result['validated_count']}")
            print(f"  Rejected readings: {result['rejected_count']}")
            print(f"  Deduplicated: {result['deduplicated_count']}")
            print(f"  Data reduction: {result['reduction_pct']:.1f}%")
            
            if result['validation_stats']:
                print(f"\nValidation reasons:")
                for reason, count in result['validation_stats'].get('reasons', {}).items():
                    print(f"  - {reason}: {count}")
            
            if result['multimodal_detection']['is_multimodal']:
                print(f"\nMultimodal detection: YES")
                print(f"  Detected {result['multimodal_detection']['num_clusters']} distinct weight clusters")
                if 'virtual_users' in result:
                    for vid, stats in result['virtual_users'].items():
                        print(f"  - {vid}: {stats['count']} readings, mean={stats['mean']:.1f}kg")
            else:
                print(f"\nMultimodal detection: NO ({result['multimodal_detection'].get('reason', 'Single user')})")
            
            if result.get('final_stats'):
                stats = result['final_stats']
                print(f"\nCleaned data statistics:")
                print(f"  Mean weight: {stats['mean_weight']:.1f} kg")
                print(f"  Std deviation: {stats['std_deviation']:.1f} kg")
                print(f"  Weight range: {stats['weight_range']:.1f} kg")
                print(f"  Coefficient of variation: {stats['cv']:.2f}")
                print(f"  Rapid changes remaining: {stats['rapid_changes_remaining']}")
        
        output_path = Path('output/benchmark_after_fixes.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"Validation results saved to: {output_path}")
        print(f"{'='*80}")
        
        self._print_comparison()
    
    def _print_comparison(self):
        """Print before/after comparison"""
        print("\n" + "="*80)
        print("COMPARISON: BEFORE vs AFTER FIXES")
        print("="*80)
        
        before_path = Path('output/benchmark_before_fixes.json')
        if before_path.exists():
            with open(before_path, 'r') as f:
                before = json.load(f)
        else:
            print("No before benchmark found. Run test_data_quality_issues.py first.")
            return
        
        comparisons = {
            'madness_case': {
                'before_rejected': before.get('madness_case_deviations', {}).get('would_be_rejected', 0),
                'after_rejected': self.results.get('madness_case', {}).get('rejected_count', 0),
                'before_rate': before.get('madness_case_deviations', {}).get('rejection_rate', 0),
                'after_rate': self.results.get('madness_case', {}).get('reduction_pct', 0)
            },
            'multi_person': {
                'before_clusters': before.get('multi_person_multimodal', {}).get('would_split_into', 1),
                'after_clusters': self.results.get('multi_person', {}).get('multimodal_detection', {}).get('num_clusters', 1)
            },
            'outliers_same': {
                'before_reduction': before.get('outliers_same_integrity', {}).get('reduction_pct', 0),
                'after_reduction': self.results.get('outliers_same', {}).get('reduction_pct', 0),
                'before_final': before.get('outliers_same_integrity', {}).get('final_reading_count', 0),
                'after_final': self.results.get('outliers_same', {}).get('validated_count', 0)
            }
        }
        
        print("\n1. MADNESS CASE (Extreme deviations):")
        print(f"   Rejection rate: {comparisons['madness_case']['before_rate']:.1f}% → {comparisons['madness_case']['after_rate']:.1f}%")
        print(f"   Rejected readings: {comparisons['madness_case']['before_rejected']} → {comparisons['madness_case']['after_rejected']}")
        
        print("\n2. MULTI-PERSON (Account sharing):")
        print(f"   Detected clusters: {comparisons['multi_person']['before_clusters']} → {comparisons['multi_person']['after_clusters']}")
        
        print("\n3. DATA INTEGRITY (Future dates & duplicates):")
        print(f"   Data reduction: {comparisons['outliers_same']['before_reduction']:.1f}% → {comparisons['outliers_same']['after_reduction']:.1f}%")
        print(f"   Final count: {comparisons['outliers_same']['before_final']} → {comparisons['outliers_same']['after_final']}")

if __name__ == "__main__":
    processor = ImprovedProcessor()
    processor.run_validation()