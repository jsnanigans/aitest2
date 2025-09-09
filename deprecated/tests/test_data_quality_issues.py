#!/usr/bin/env python3
"""
Test benchmarks to highlight and confirm data quality issues in the three challenging users.
This establishes baseline metrics before implementing fixes.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

class DataQualityBenchmark:
    def __init__(self, csv_path='./2025-09-05_optimized.csv'):
        self.csv_path = csv_path
        self.test_users = {
            '0040872d-333a-4ace-8c5a-b2fcd056e65a': 'madness_case',
            '0675ed39-53be-480b-baa6-fa53fc33709f': 'multi_person', 
            '055b0c48-d5b4-44cb-8772-48154999e6c3': 'outliers_same'
        }
        self.results = {}
        
    def load_user_data(self, user_id):
        """Load data for a specific user"""
        readings = []
        with open(self.csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5 and parts[0] == user_id:
                    readings.append({
                        'date': datetime.fromisoformat(parts[1].replace(' ', 'T')),
                        'source': parts[2],
                        'weight': float(parts[3]),
                        'unit': parts[4]
                    })
        return sorted(readings, key=lambda x: x['date'])
    
    def test_extreme_deviations(self, user_id):
        """Test 1: Identify extreme weight deviations (madness case)"""
        readings = self.load_user_data(user_id)
        if not readings:
            return None
            
        weights = [r['weight'] for r in readings]
        
        # Calculate baseline metrics
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        # Find rapid changes (>15% in 24 hours)
        rapid_changes = []
        extreme_outliers = []
        
        for i in range(1, len(readings)):
            time_diff = (readings[i]['date'] - readings[i-1]['date']).total_seconds() / 3600
            weight_diff = abs(readings[i]['weight'] - readings[i-1]['weight'])
            pct_change = weight_diff / readings[i-1]['weight'] * 100
            
            if time_diff <= 24 and pct_change > 15:
                rapid_changes.append({
                    'from': readings[i-1],
                    'to': readings[i],
                    'change_pct': pct_change,
                    'hours': time_diff
                })
        
        # Find statistical outliers (>2.5 std)
        for i, weight in enumerate(weights):
            z_score = abs((weight - mean_weight) / std_weight) if std_weight > 0 else 0
            if z_score > 2.5:
                extreme_outliers.append({
                    'reading': readings[i],
                    'z_score': z_score,
                    'deviation_pct': abs((weight - mean_weight) / mean_weight * 100)
                })
        
        return {
            'user_id': user_id,
            'test': 'extreme_deviations',
            'total_readings': len(readings),
            'mean_weight': mean_weight,
            'std_deviation': std_weight,
            'weight_range': max(weights) - min(weights),
            'rapid_changes_count': len(rapid_changes),
            'extreme_outliers_count': len(extreme_outliers),
            'would_be_rejected': len(rapid_changes) + len(extreme_outliers),
            'rejection_rate': (len(rapid_changes) + len(extreme_outliers)) / len(readings) * 100,
            'sample_rapid_changes': rapid_changes[:3],
            'sample_outliers': extreme_outliers[:3]
        }
    
    def test_multimodal_distribution(self, user_id):
        """Test 2: Detect multimodal weight distribution (multiple people)"""
        readings = self.load_user_data(user_id)
        if not readings:
            return None
            
        weights = np.array([r['weight'] for r in readings])
        
        # Simple clustering approach: find gaps > 15kg
        sorted_weights = np.sort(weights)
        clusters = []
        current_cluster = [sorted_weights[0]]
        
        for i in range(1, len(sorted_weights)):
            if sorted_weights[i] - current_cluster[-1] > 15:
                clusters.append(current_cluster)
                current_cluster = [sorted_weights[i]]
            else:
                current_cluster.append(sorted_weights[i])
        clusters.append(current_cluster)
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster in clusters:
            if len(cluster) >= 3:  # Minimum cluster size
                cluster_stats.append({
                    'mean': np.mean(cluster),
                    'std': np.std(cluster),
                    'count': len(cluster),
                    'range': (min(cluster), max(cluster))
                })
        
        # Detect if multimodal
        is_multimodal = len(cluster_stats) > 1
        
        # Map readings to clusters
        if is_multimodal:
            reading_clusters = []
            for r in readings:
                # Find closest cluster
                min_dist = float('inf')
                assigned_cluster = 0
                for idx, cs in enumerate(cluster_stats):
                    dist = abs(r['weight'] - cs['mean'])
                    if dist < min_dist:
                        min_dist = dist
                        assigned_cluster = idx
                reading_clusters.append(assigned_cluster)
        else:
            reading_clusters = [0] * len(readings)
        
        return {
            'user_id': user_id,
            'test': 'multimodal_distribution',
            'total_readings': len(readings),
            'is_multimodal': is_multimodal,
            'num_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats,
            'would_split_into': len(cluster_stats) if is_multimodal else 1,
            'weight_distribution': {
                'min': float(min(weights)),
                'p25': float(np.percentile(weights, 25)),
                'median': float(np.median(weights)),
                'p75': float(np.percentile(weights, 75)),
                'max': float(max(weights))
            }
        }
    
    def test_data_integrity(self, user_id):
        """Test 3: Check data integrity issues (future dates, duplicates)"""
        readings = self.load_user_data(user_id)
        if not readings:
            return None
            
        now = datetime.now()
        future_readings = []
        duplicate_same_day = defaultdict(list)
        precision_anomalies = []
        
        for r in readings:
            # Check future dates
            if r['date'] > now:
                future_readings.append(r)
            
            # Group by day for duplicate detection
            day_key = r['date'].date()
            duplicate_same_day[day_key].append(r)
            
            # Check precision anomalies
            decimal_places = len(str(r['weight']).split('.')[-1]) if '.' in str(r['weight']) else 0
            if decimal_places > 2:
                precision_anomalies.append({
                    'date': r['date'].isoformat(),
                    'weight': r['weight'],
                    'decimal_places': decimal_places
                })
        
        # Find days with near-duplicate weights
        duplicate_days = {}
        for day, day_readings in duplicate_same_day.items():
            if len(day_readings) > 1:
                weights = [r['weight'] for r in day_readings]
                # Check if weights are within 0.5kg
                if max(weights) - min(weights) < 0.5:
                    duplicate_days[day.isoformat()] = {
                        'count': len(day_readings),
                        'weights': weights,
                        'would_dedupe_to': 1
                    }
        
        return {
            'user_id': user_id,
            'test': 'data_integrity',
            'total_readings': len(readings),
            'future_readings_count': len(future_readings),
            'duplicate_days_count': len(duplicate_days),
            'precision_anomalies_count': len(precision_anomalies),
            'would_be_rejected': len(future_readings),
            'would_be_deduped': sum(d['count'] - 1 for d in duplicate_days.values()),
            'final_reading_count': len(readings) - len(future_readings) - sum(d['count'] - 1 for d in duplicate_days.values()),
            'reduction_pct': ((len(future_readings) + sum(d['count'] - 1 for d in duplicate_days.values())) / len(readings) * 100),
            'sample_future': [r['date'].isoformat() for r in future_readings[:5]],
            'sample_duplicates': dict(list(duplicate_days.items())[:5])
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks and generate report"""
        print("="*80)
        print("DATA QUALITY BENCHMARKS - BEFORE FIXES")
        print("="*80)
        
        for user_id, label in self.test_users.items():
            print(f"\n{'-'*80}")
            print(f"USER: {user_id}")
            print(f"LABEL: {label.upper()}")
            print(f"{'-'*80}")
            
            # Run appropriate tests
            if label == 'madness_case':
                result = self.test_extreme_deviations(user_id)
                if result:
                    print(f"\nTest 1: Extreme Deviations")
                    print(f"  Total readings: {result['total_readings']}")
                    print(f"  Mean weight: {result['mean_weight']:.1f} kg")
                    print(f"  Std deviation: {result['std_deviation']:.1f} kg")
                    print(f"  Weight range: {result['weight_range']:.1f} kg")
                    print(f"  Rapid changes (>15% in 24h): {result['rapid_changes_count']}")
                    print(f"  Extreme outliers (>2.5 std): {result['extreme_outliers_count']}")
                    print(f"  WOULD BE REJECTED: {result['would_be_rejected']} readings ({result['rejection_rate']:.1f}%)")
                    self.results[f"{label}_deviations"] = result
            
            elif label == 'multi_person':
                result = self.test_multimodal_distribution(user_id)
                if result:
                    print(f"\nTest 2: Multimodal Distribution")
                    print(f"  Total readings: {result['total_readings']}")
                    print(f"  Is multimodal: {result['is_multimodal']}")
                    print(f"  Number of clusters detected: {result['num_clusters']}")
                    if result['cluster_stats']:
                        for i, cluster in enumerate(result['cluster_stats']):
                            print(f"  Cluster {i+1}: mean={cluster['mean']:.1f}kg, count={cluster['count']}, range={cluster['range'][0]:.1f}-{cluster['range'][1]:.1f}kg")
                    print(f"  WOULD SPLIT INTO: {result['would_split_into']} separate tracking streams")
                    self.results[f"{label}_multimodal"] = result
            
            elif label == 'outliers_same':
                result = self.test_data_integrity(user_id)
                if result:
                    print(f"\nTest 3: Data Integrity")
                    print(f"  Total readings: {result['total_readings']}")
                    print(f"  Future date readings: {result['future_readings_count']}")
                    print(f"  Days with duplicates: {result['duplicate_days_count']}")
                    print(f"  Precision anomalies: {result['precision_anomalies_count']}")
                    print(f"  WOULD BE REJECTED: {result['would_be_rejected']} readings")
                    print(f"  WOULD BE DEDUPED: {result['would_be_deduped']} readings")
                    print(f"  FINAL COUNT: {result['final_reading_count']} (reduction: {result['reduction_pct']:.1f}%)")
                    self.results[f"{label}_integrity"] = result
        
        # Save benchmark results
        output_path = Path('output/benchmark_before_fixes.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"Benchmark results saved to: {output_path}")
        print(f"{'='*80}")
        
        return self.results

if __name__ == "__main__":
    benchmark = DataQualityBenchmark()
    benchmark.run_all_benchmarks()