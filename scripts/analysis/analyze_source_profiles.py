"""
Comprehensive source type profiling and analysis.
Generates detailed report on each source type's characteristics.
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.processor import WeightProcessor
from src.database import ProcessorStateDB, get_state_db


class SourceProfiler:
    """Comprehensive profiler for each source type."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.source_profiles = defaultdict(lambda: {
            'measurements': [],
            'timestamps': [],
            'users': defaultdict(list),
            'weight_distribution': [],
            'time_gaps': [],
            'daily_patterns': defaultdict(int),
            'weekly_patterns': defaultdict(int),
            'quality_metrics': {
                'valid_count': 0,
                'invalid_count': 0,
                'outlier_count': 0,
                'duplicate_count': 0,
                'round_numbers': 0,
                'decimal_precision': [],
                'weight_changes': [],
                'consistency_scores': []
            },
            'user_behavior': {
                'single_source_users': set(),
                'multi_source_users': set(),
                'measurement_frequency': [],
                'user_retention': []
            }
        })
        
    def load_and_profile(self, sample_size: int = None):
        """Load data and create comprehensive profiles."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE SOURCE TYPE PROFILING")
        print(f"{'='*80}\n")
        
        # First pass: collect all data
        row_count = 0
        user_sources = defaultdict(set)
        user_first_seen = {}
        user_last_seen = {}
        
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                row_count += 1
                if sample_size and row_count > sample_size:
                    break
                
                if not row['weight'] or not row['effectiveDateTime']:
                    continue
                
                try:
                    weight = float(row['weight'])
                    timestamp = datetime.fromisoformat(row['effectiveDateTime'])
                    source = row['source_type']
                    user_id = row['user_id']
                    
                    # Track user-source relationships
                    user_sources[user_id].add(source)
                    
                    # Track user activity span
                    if user_id not in user_first_seen:
                        user_first_seen[user_id] = timestamp
                        user_last_seen[user_id] = timestamp
                    else:
                        user_first_seen[user_id] = min(user_first_seen[user_id], timestamp)
                        user_last_seen[user_id] = max(user_last_seen[user_id], timestamp)
                    
                    # Profile the source
                    profile = self.source_profiles[source]
                    
                    # Basic measurements
                    profile['measurements'].append(weight)
                    profile['timestamps'].append(timestamp)
                    profile['users'][user_id].append({
                        'weight': weight,
                        'timestamp': timestamp
                    })
                    
                    # Quality metrics
                    if 30 <= weight <= 400:
                        profile['quality_metrics']['valid_count'] += 1
                    else:
                        profile['quality_metrics']['invalid_count'] += 1
                    
                    # Check for round numbers
                    if weight % 5 == 0:
                        profile['quality_metrics']['round_numbers'] += 1
                    
                    # Track decimal precision
                    decimal_places = len(str(weight).split('.')[-1]) if '.' in str(weight) else 0
                    profile['quality_metrics']['decimal_precision'].append(decimal_places)
                    
                    # Time patterns
                    profile['daily_patterns'][timestamp.hour] += 1
                    profile['weekly_patterns'][timestamp.weekday()] += 1
                    
                except (ValueError, TypeError) as e:
                    continue
        
        print(f"Processed {row_count:,} rows")
        print(f"Found {len(self.source_profiles)} unique source types")
        print(f"Tracking {len(user_sources):,} users\n")
        
        # Second pass: calculate derived metrics
        self._calculate_derived_metrics(user_sources, user_first_seen, user_last_seen)
        
        return self.source_profiles
    
    def _calculate_derived_metrics(self, user_sources, user_first_seen, user_last_seen):
        """Calculate derived metrics for each source."""
        
        for source, profile in self.source_profiles.items():
            # User behavior analysis
            for user_id in profile['users']:
                if len(user_sources[user_id]) == 1:
                    profile['user_behavior']['single_source_users'].add(user_id)
                else:
                    profile['user_behavior']['multi_source_users'].add(user_id)
                
                # Calculate measurement frequency
                user_data = profile['users'][user_id]
                if len(user_data) > 1:
                    user_data.sort(key=lambda x: x['timestamp'])
                    
                    # Time gaps between measurements
                    for i in range(1, len(user_data)):
                        gap = (user_data[i]['timestamp'] - user_data[i-1]['timestamp']).days
                        profile['time_gaps'].append(gap)
                    
                    # Weight changes
                    for i in range(1, len(user_data)):
                        change = abs(user_data[i]['weight'] - user_data[i-1]['weight'])
                        profile['quality_metrics']['weight_changes'].append(change)
                        
                        # Check for duplicates
                        if user_data[i]['weight'] == user_data[i-1]['weight']:
                            profile['quality_metrics']['duplicate_count'] += 1
                    
                    # User retention (days active)
                    if user_id in user_first_seen and user_id in user_last_seen:
                        retention_days = (user_last_seen[user_id] - user_first_seen[user_id]).days
                        profile['user_behavior']['user_retention'].append(retention_days)
                    
                    # Measurement frequency (measurements per week)
                    if retention_days > 0:
                        freq = len(user_data) / (retention_days / 7.0)
                        profile['user_behavior']['measurement_frequency'].append(freq)
            
            # Calculate consistency score for each user
            for user_id, user_data in profile['users'].items():
                if len(user_data) > 2:
                    weights = [d['weight'] for d in user_data]
                    # Consistency = inverse of coefficient of variation
                    if np.mean(weights) > 0:
                        cv = np.std(weights) / np.mean(weights)
                        consistency = 1 / (1 + cv)
                        profile['quality_metrics']['consistency_scores'].append(consistency)
    
    def generate_report(self):
        """Generate comprehensive report for each source type."""
        
        print(f"\n{'='*80}")
        print("DETAILED SOURCE TYPE PROFILES")
        print(f"{'='*80}\n")
        
        # Sort sources by measurement count
        sorted_sources = sorted(self.source_profiles.items(), 
                              key=lambda x: len(x[1]['measurements']), 
                              reverse=True)
        
        for rank, (source, profile) in enumerate(sorted_sources, 1):
            self._print_source_profile(rank, source, profile)
        
        # Generate comparative analysis
        self._generate_comparative_analysis()
        
        # Generate visualizations
        self._create_visualizations()
    
    def _print_source_profile(self, rank: int, source: str, profile: Dict):
        """Print detailed profile for a single source."""
        
        measurements = profile['measurements']
        if not measurements:
            return
        
        print(f"\n{'='*60}")
        print(f"#{rank}: {source}")
        print(f"{'='*60}")
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS")
        print(f"  Total measurements: {len(measurements):,}")
        print(f"  Unique users: {len(profile['users']):,}")
        print(f"  Date range: {min(profile['timestamps']).date()} to {max(profile['timestamps']).date()}")
        
        # User composition
        single_users = len(profile['user_behavior']['single_source_users'])
        multi_users = len(profile['user_behavior']['multi_source_users'])
        print(f"\n👥 USER COMPOSITION")
        print(f"  Single-source users: {single_users:,} ({single_users/(single_users+multi_users)*100:.1f}%)")
        print(f"  Multi-source users: {multi_users:,} ({multi_users/(single_users+multi_users)*100:.1f}%)")
        
        if profile['user_behavior']['measurement_frequency']:
            print(f"  Avg measurements/week/user: {np.mean(profile['user_behavior']['measurement_frequency']):.1f}")
        
        if profile['user_behavior']['user_retention']:
            print(f"  Avg user retention: {np.mean(profile['user_behavior']['user_retention']):.0f} days")
        
        # Weight distribution
        valid_weights = [w for w in measurements if 30 <= w <= 400]
        if valid_weights:
            print(f"\n⚖️ WEIGHT DISTRIBUTION")
            print(f"  Mean: {np.mean(valid_weights):.1f} kg")
            print(f"  Median: {np.median(valid_weights):.1f} kg")
            print(f"  Std Dev: {np.std(valid_weights):.1f} kg")
            print(f"  Range: {min(valid_weights):.1f} - {max(valid_weights):.1f} kg")
            print(f"  25th percentile: {np.percentile(valid_weights, 25):.1f} kg")
            print(f"  75th percentile: {np.percentile(valid_weights, 75):.1f} kg")
        
        # Data quality
        quality = profile['quality_metrics']
        total = quality['valid_count'] + quality['invalid_count']
        
        print(f"\n✅ DATA QUALITY")
        print(f"  Valid measurements: {quality['valid_count']:,} ({quality['valid_count']/total*100:.1f}%)")
        print(f"  Invalid measurements: {quality['invalid_count']:,} ({quality['invalid_count']/total*100:.1f}%)")
        
        if quality['decimal_precision']:
            avg_precision = np.mean(quality['decimal_precision'])
            print(f"  Avg decimal precision: {avg_precision:.1f} places")
        
        print(f"  Round numbers (÷5): {quality['round_numbers']:,} ({quality['round_numbers']/total*100:.1f}%)")
        print(f"  Duplicate consecutive: {quality['duplicate_count']:,} ({quality['duplicate_count']/total*100:.1f}%)")
        
        if quality['weight_changes']:
            changes = quality['weight_changes']
            print(f"\n📈 WEIGHT CHANGE PATTERNS")
            print(f"  Avg change between measurements: {np.mean(changes):.2f} kg")
            print(f"  Median change: {np.median(changes):.2f} kg")
            print(f"  Max single change: {max(changes):.1f} kg")
            print(f"  Changes >5kg: {sum(1 for c in changes if c > 5):,} ({sum(1 for c in changes if c > 5)/len(changes)*100:.1f}%)")
            print(f"  Changes >10kg: {sum(1 for c in changes if c > 10):,} ({sum(1 for c in changes if c > 10)/len(changes)*100:.1f}%)")
        
        if quality['consistency_scores']:
            print(f"  Avg consistency score: {np.mean(quality['consistency_scores']):.2f} (0=variable, 1=consistent)")
        
        # Time patterns
        if profile['time_gaps']:
            print(f"\n⏰ TEMPORAL PATTERNS")
            print(f"  Avg days between measurements: {np.mean(profile['time_gaps']):.1f}")
            print(f"  Median days between: {np.median(profile['time_gaps']):.0f}")
            
            # Find most active time of day
            if profile['daily_patterns']:
                peak_hour = max(profile['daily_patterns'].items(), key=lambda x: x[1])
                print(f"  Peak activity hour: {peak_hour[0]:02d}:00 ({peak_hour[1]:,} measurements)")
            
            # Find most active day of week
            if profile['weekly_patterns']:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                peak_day = max(profile['weekly_patterns'].items(), key=lambda x: x[1])
                print(f"  Peak activity day: {days[peak_day[0]]} ({peak_day[1]:,} measurements)")
        
        # Reliability assessment
        print(f"\n🎯 RELIABILITY ASSESSMENT")
        reliability_score = self._calculate_reliability_score(profile)
        print(f"  Overall reliability score: {reliability_score:.1f}/10")
        
        if reliability_score >= 8:
            print(f"  Rating: ⭐⭐⭐⭐⭐ EXCELLENT - Highly reliable source")
        elif reliability_score >= 6:
            print(f"  Rating: ⭐⭐⭐⭐ GOOD - Generally reliable")
        elif reliability_score >= 4:
            print(f"  Rating: ⭐⭐⭐ MODERATE - Some concerns")
        else:
            print(f"  Rating: ⭐⭐ POOR - Significant reliability issues")
    
    def _calculate_reliability_score(self, profile: Dict) -> float:
        """Calculate reliability score (0-10) for a source."""
        score = 10.0
        
        quality = profile['quality_metrics']
        total = quality['valid_count'] + quality['invalid_count']
        
        if total == 0:
            return 0.0
        
        # Penalize invalid measurements
        invalid_rate = quality['invalid_count'] / total
        score -= invalid_rate * 3
        
        # Penalize high round number rate (suggests estimation)
        round_rate = quality['round_numbers'] / total
        if round_rate > 0.5:
            score -= 2
        elif round_rate > 0.3:
            score -= 1
        
        # Penalize high duplicate rate
        dup_rate = quality['duplicate_count'] / total
        if dup_rate > 0.3:
            score -= 1.5
        elif dup_rate > 0.15:
            score -= 0.5
        
        # Penalize extreme weight changes
        if quality['weight_changes']:
            extreme_changes = sum(1 for c in quality['weight_changes'] if c > 10)
            extreme_rate = extreme_changes / len(quality['weight_changes'])
            score -= extreme_rate * 2
        
        # Reward consistency
        if quality['consistency_scores']:
            avg_consistency = np.mean(quality['consistency_scores'])
            score += avg_consistency * 1  # Bonus for consistency
        
        return max(0, min(10, score))
    
    def _generate_comparative_analysis(self):
        """Generate comparative analysis across all sources."""
        
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*80}\n")
        
        # Prepare comparison data
        comparison = []
        for source, profile in self.source_profiles.items():
            if len(profile['measurements']) < 100:  # Skip small sources
                continue
            
            quality = profile['quality_metrics']
            total = quality['valid_count'] + quality['invalid_count']
            
            valid_weights = [w for w in profile['measurements'] if 30 <= w <= 400]
            
            comparison.append({
                'source': source[:30],  # Truncate for display
                'count': len(profile['measurements']),
                'users': len(profile['users']),
                'mean_weight': np.mean(valid_weights) if valid_weights else 0,
                'std_weight': np.std(valid_weights) if valid_weights else 0,
                'invalid_rate': quality['invalid_count'] / total if total > 0 else 0,
                'round_rate': quality['round_numbers'] / total if total > 0 else 0,
                'dup_rate': quality['duplicate_count'] / total if total > 0 else 0,
                'reliability': self._calculate_reliability_score(profile)
            })
        
        # Sort by reliability
        comparison.sort(key=lambda x: x['reliability'], reverse=True)
        
        print("📊 RELIABILITY RANKING")
        print(f"{'Rank':<5} {'Source':<30} {'Score':<7} {'Count':<10} {'Users':<8} {'Issues'}")
        print("-" * 80)
        
        for i, comp in enumerate(comparison, 1):
            issues = []
            if comp['invalid_rate'] > 0.01:
                issues.append(f"invalid:{comp['invalid_rate']:.1%}")
            if comp['round_rate'] > 0.3:
                issues.append(f"round:{comp['round_rate']:.0%}")
            if comp['dup_rate'] > 0.15:
                issues.append(f"dup:{comp['dup_rate']:.0%}")
            
            print(f"{i:<5} {comp['source']:<30} {comp['reliability']:<6.1f} "
                  f"{comp['count']:<10,} {comp['users']:<8,} {', '.join(issues)}")
        
        # Statistical comparison
        print(f"\n📈 STATISTICAL COMPARISON")
        print(f"{'Source':<30} {'Mean±SD (kg)':<20} {'CV':<8}")
        print("-" * 60)
        
        for comp in comparison[:10]:  # Top 10
            cv = comp['std_weight'] / comp['mean_weight'] if comp['mean_weight'] > 0 else 0
            print(f"{comp['source']:<30} {comp['mean_weight']:.1f}±{comp['std_weight']:.1f}"
                  f"{' '*5}{cv:.3f}")
    
    def _create_visualizations(self):
        """Create visualization plots."""
        
        # Prepare data for visualization
        sources = []
        reliabilities = []
        counts = []
        
        for source, profile in self.source_profiles.items():
            if len(profile['measurements']) >= 100:
                sources.append(source[:20])  # Truncate for display
                reliabilities.append(self._calculate_reliability_score(profile))
                counts.append(len(profile['measurements']))
        
        if not sources:
            return
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Reliability scores
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['green' if r >= 7 else 'orange' if r >= 4 else 'red' for r in reliabilities]
        bars = ax1.barh(range(len(sources)), reliabilities, color=colors)
        ax1.set_yticks(range(len(sources)))
        ax1.set_yticklabels(sources)
        ax1.set_xlabel('Reliability Score (0-10)')
        ax1.set_title('Source Reliability Assessment')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, reliabilities)):
            ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}', va='center')
        
        # Plot 2: Measurement counts (log scale)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.barh(range(len(sources)), counts, color='steelblue')
        ax2.set_yticks(range(len(sources)))
        ax2.set_yticklabels(sources)
        ax2.set_xlabel('Number of Measurements (log scale)')
        ax2.set_xscale('log')
        ax2.set_title('Data Volume by Source')
        ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: Weight distributions
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get weight distributions for top sources
        top_sources = sorted(self.source_profiles.items(), 
                           key=lambda x: len(x[1]['measurements']), 
                           reverse=True)[:5]
        
        positions = []
        labels = []
        for i, (source, profile) in enumerate(top_sources):
            valid_weights = [w for w in profile['measurements'] if 30 <= w <= 400]
            if valid_weights:
                bp = ax3.boxplot(valid_weights, positions=[i], widths=0.6,
                                patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(f'C{i}')
                labels.append(source[:25])
        
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Weight (kg)')
        ax3.set_title('Weight Distribution by Top 5 Sources')
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'source_profiles_analysis.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_file}")
        plt.close()
    
    def save_json_report(self, output_file: str = 'source_profiles.json'):
        """Save detailed profiles as JSON."""
        
        # Convert sets and datetime objects to serializable format
        serializable_profiles = {}
        
        for source, profile in self.source_profiles.items():
            serializable_profiles[source] = {
                'measurement_count': len(profile['measurements']),
                'user_count': len(profile['users']),
                'weight_stats': {
                    'mean': float(np.mean(profile['measurements'])) if profile['measurements'] else 0,
                    'median': float(np.median(profile['measurements'])) if profile['measurements'] else 0,
                    'std': float(np.std(profile['measurements'])) if profile['measurements'] else 0,
                    'min': float(min(profile['measurements'])) if profile['measurements'] else 0,
                    'max': float(max(profile['measurements'])) if profile['measurements'] else 0
                },
                'quality_metrics': {
                    'valid_count': profile['quality_metrics']['valid_count'],
                    'invalid_count': profile['quality_metrics']['invalid_count'],
                    'round_numbers': profile['quality_metrics']['round_numbers'],
                    'duplicate_count': profile['quality_metrics']['duplicate_count'],
                    'avg_decimal_precision': float(np.mean(profile['quality_metrics']['decimal_precision'])) 
                                           if profile['quality_metrics']['decimal_precision'] else 0
                },
                'reliability_score': float(self._calculate_reliability_score(profile)),
                'user_behavior': {
                    'single_source_users': len(profile['user_behavior']['single_source_users']),
                    'multi_source_users': len(profile['user_behavior']['multi_source_users']),
                    'avg_measurements_per_week': float(np.mean(profile['user_behavior']['measurement_frequency'])) 
                                                if profile['user_behavior']['measurement_frequency'] else 0,
                    'avg_retention_days': float(np.mean(profile['user_behavior']['user_retention'])) 
                                        if profile['user_behavior']['user_retention'] else 0
                }
            }
        
        output_path = os.path.join('output', output_file)
        with open(output_path, 'w') as f:
            json.dump(serializable_profiles, f, indent=2)
        
        print(f"📄 JSON report saved to: {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_source_profiles.py <csv_file> [sample_size]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Create profiler and run analysis
    profiler = SourceProfiler(data_file)
    profiler.load_and_profile(sample_size)
    profiler.generate_report()
    profiler.save_json_report()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
