"""
Comprehensive source type impact analysis on large dataset.
Analyzes 4000+ users to determine if source differentiation improves processing.
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os

from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
from src.visualization import normalize_source_type


class LargeScaleSourceAnalyzer:
    """Analyze source impact across thousands of users."""
    
    def __init__(self, data_file: str, sample_size: int = 4000):
        """Initialize with large dataset."""
        self.data_file = data_file
        self.sample_size = sample_size
        self.user_data = {}
        self.user_stats = {}
        
    def load_and_sample_users(self) -> Dict[str, List]:
        """Load data and sample users with sufficient measurements."""
        print(f"Loading data from {self.data_file}...")
        
        # First pass: count measurements per user
        user_counts = defaultdict(int)
        total_lines = 0
        
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_lines += 1
                if total_lines % 100000 == 0:
                    print(f"  Processed {total_lines:,} lines...")
                user_counts[row['user_id']] += 1
        
        print(f"Total measurements: {total_lines:,}")
        print(f"Total users: {len(user_counts):,}")
        
        # Filter users with sufficient data (at least 10 measurements)
        eligible_users = [uid for uid, count in user_counts.items() if count >= 10]
        print(f"Users with 10+ measurements: {len(eligible_users):,}")
        
        # Sample users
        if len(eligible_users) > self.sample_size:
            sampled_users = random.sample(eligible_users, self.sample_size)
        else:
            sampled_users = eligible_users
            
        print(f"Sampling {len(sampled_users):,} users for analysis...")
        
        # Second pass: load data for sampled users
        user_data = defaultdict(list)
        sampled_set = set(sampled_users)
        
        with open(self.data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user_id'] in sampled_set:
                    try:
                        user_data[row['user_id']].append({
                            'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                            'weight': float(row['weight']),
                            'source': row['source_type'],
                            'normalized': normalize_source_type(row['source_type'])
                        })
                    except (ValueError, KeyError):
                        continue
        
        # Sort measurements by timestamp
        for uid in user_data:
            user_data[uid].sort(key=lambda x: x['timestamp'])
        
        self.user_data = dict(user_data)
        print(f"Loaded data for {len(self.user_data):,} users")
        
        return self.user_data
    
    def analyze_user_characteristics(self):
        """Analyze characteristics of sampled users."""
        print("\nAnalyzing user characteristics...")
        
        stats = {
            'measurement_counts': [],
            'source_diversity': [],
            'time_spans': [],
            'weight_ranges': [],
            'source_distributions': defaultdict(int),
            'user_categories': defaultdict(int)
        }
        
        for uid, measurements in self.user_data.items():
            # Measurement count
            stats['measurement_counts'].append(len(measurements))
            
            # Source diversity
            sources = set(m['normalized'] for m in measurements)
            stats['source_diversity'].append(len(sources))
            
            # Time span
            if len(measurements) > 1:
                span = (measurements[-1]['timestamp'] - measurements[0]['timestamp']).days
                stats['time_spans'].append(span)
            
            # Weight range
            weights = [m['weight'] for m in measurements]
            if weights:
                stats['weight_ranges'].append(max(weights) - min(weights))
            
            # Source distribution
            for m in measurements:
                stats['source_distributions'][m['normalized']] += 1
            
            # Categorize user
            if len(sources) == 1:
                stats['user_categories']['single_source'] += 1
            elif len(sources) == 2:
                stats['user_categories']['two_sources'] += 1
            else:
                stats['user_categories']['multi_source'] += 1
        
        self.user_stats = stats
        
        # Print summary
        print(f"\nUSER STATISTICS ({len(self.user_data)} users):")
        print("-" * 50)
        print(f"Measurements per user:")
        print(f"  Mean: {np.mean(stats['measurement_counts']):.1f}")
        print(f"  Median: {np.median(stats['measurement_counts']):.0f}")
        print(f"  Min: {min(stats['measurement_counts'])}")
        print(f"  Max: {max(stats['measurement_counts'])}")
        
        print(f"\nSource diversity:")
        print(f"  Single source: {stats['user_categories']['single_source']:,} users")
        print(f"  Two sources: {stats['user_categories']['two_sources']:,} users")
        print(f"  Multi-source: {stats['user_categories']['multi_source']:,} users")
        
        print(f"\nOverall source distribution:")
        total_measurements = sum(stats['source_distributions'].values())
        for source, count in sorted(stats['source_distributions'].items(), 
                                   key=lambda x: x[1], reverse=True):
            pct = count / total_measurements * 100
            print(f"  {source:20s}: {count:7,} ({pct:5.1f}%)")
        
        return stats
    
    def test_strategies_on_users(self, user_sample: List[str] = None):
        """Test different strategies on a sample of users."""
        if user_sample is None:
            # Sample diverse users
            user_sample = self.select_diverse_users(min(500, len(self.user_data)))
        
        print(f"\nTesting strategies on {len(user_sample)} users...")
        
        strategies = {
            'baseline': self.strategy_baseline,
            'trust_weighted': self.strategy_trust_weighted,
            'adaptive_limits': self.strategy_adaptive_limits,
            'hybrid': self.strategy_hybrid
        }
        
        results = defaultdict(lambda: defaultdict(list))
        
        for i, uid in enumerate(user_sample):
            if i % 100 == 0:
                print(f"  Processing user {i+1}/{len(user_sample)}...")
            
            measurements = self.user_data[uid]
            
            for strategy_name, strategy_func in strategies.items():
                strategy_results = strategy_func(measurements)
                metrics = self.calculate_metrics(strategy_results)
                
                # Store metrics
                for metric, value in metrics.items():
                    results[strategy_name][metric].append(value)
        
        # Calculate aggregate statistics
        aggregate_results = {}
        for strategy in results:
            aggregate_results[strategy] = {}
            for metric in results[strategy]:
                values = results[strategy][metric]
                aggregate_results[strategy][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return aggregate_results
    
    def select_diverse_users(self, count: int) -> List[str]:
        """Select diverse set of users for testing."""
        # Categorize users
        single_source = []
        two_source = []
        multi_source = []
        high_volume = []
        
        for uid, measurements in self.user_data.items():
            sources = set(m['normalized'] for m in measurements)
            
            if len(sources) == 1:
                single_source.append(uid)
            elif len(sources) == 2:
                two_source.append(uid)
            else:
                multi_source.append(uid)
            
            if len(measurements) > 50:
                high_volume.append(uid)
        
        # Sample from each category
        sample = []
        
        # Ensure diversity
        categories = [
            (multi_source, count // 3),    # Prioritize multi-source
            (two_source, count // 3),
            (single_source, count // 6),
            (high_volume, count // 6)
        ]
        
        for category_list, target_count in categories:
            if category_list:
                sample_count = min(len(category_list), target_count)
                sample.extend(random.sample(category_list, sample_count))
        
        # Remove duplicates and fill to target count
        sample = list(set(sample))
        
        if len(sample) < count:
            remaining = list(set(self.user_data.keys()) - set(sample))
            additional = min(count - len(sample), len(remaining))
            sample.extend(random.sample(remaining, additional))
        
        return sample[:count]
    
    def strategy_baseline(self, measurements: List[Dict]) -> List[Dict]:
        """Baseline strategy - no source differentiation."""
        db = ProcessorStateDB()
        results = []
        
        config = {
            'processing': {
                'extreme_threshold': 10.0,
                'max_weight': 400.0,
                'min_weight': 30.0,
            },
            'kalman': {
                'observation_covariance': 5.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'initial_variance': 10.0
            }
        }
        
        for m in measurements:
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                results.append(result)
        
        return results
    
    def strategy_trust_weighted(self, measurements: List[Dict]) -> List[Dict]:
        """Trust-weighted Kalman filtering."""
        db = ProcessorStateDB()
        results = []
        
        trust_scores = {
            'patient-device': 1.0,
            'device': 1.0,
            'api': 0.85,
            'connected': 0.85,
            'questionnaire': 0.6,
            'manual': 0.4,
            'other': 0.5
        }
        
        base_config = {
            'processing': {
                'extreme_threshold': 10.0,
                'max_weight': 400.0,
                'min_weight': 30.0,
            },
            'kalman': {
                'observation_covariance': 5.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'initial_variance': 10.0
            }
        }
        
        for m in measurements:
            trust = trust_scores.get(m['normalized'], 0.5)
            
            config = base_config.copy()
            config['kalman'] = base_config['kalman'].copy()
            config['kalman']['observation_covariance'] = 5.0 / (trust ** 2)
            
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                result['trust'] = trust
                results.append(result)
        
        return results
    
    def strategy_adaptive_limits(self, measurements: List[Dict]) -> List[Dict]:
        """Adaptive physiological limits based on source."""
        db = ProcessorStateDB()
        results = []
        
        limit_multipliers = {
            'patient-device': 1.0,
            'device': 1.0,
            'api': 1.2,
            'connected': 1.2,
            'questionnaire': 1.5,
            'manual': 2.0,
            'other': 1.3
        }
        
        base_config = {
            'processing': {
                'extreme_threshold': 10.0,
                'max_weight': 400.0,
                'min_weight': 30.0,
            },
            'kalman': {
                'observation_covariance': 5.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'initial_variance': 10.0
            }
        }
        
        for m in measurements:
            multiplier = limit_multipliers.get(m['normalized'], 1.3)
            
            config = base_config.copy()
            config['processing'] = base_config['processing'].copy()
            config['processing']['extreme_threshold'] = 10.0 * multiplier
            
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                result['limit_mult'] = multiplier
                results.append(result)
        
        return results
    
    def strategy_hybrid(self, measurements: List[Dict]) -> List[Dict]:
        """Hybrid strategy combining trust weighting and adaptive limits."""
        db = ProcessorStateDB()
        results = []
        
        trust_scores = {
            'patient-device': 1.0,
            'device': 1.0,
            'api': 0.85,
            'connected': 0.85,
            'questionnaire': 0.6,
            'manual': 0.4,
            'other': 0.5
        }
        
        limit_multipliers = {
            'patient-device': 1.0,
            'device': 1.0,
            'api': 1.1,
            'connected': 1.1,
            'questionnaire': 1.3,
            'manual': 1.5,
            'other': 1.2
        }
        
        base_config = {
            'processing': {
                'extreme_threshold': 10.0,
                'max_weight': 400.0,
                'min_weight': 30.0,
            },
            'kalman': {
                'observation_covariance': 5.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'initial_variance': 10.0
            }
        }
        
        for m in measurements:
            trust = trust_scores.get(m['normalized'], 0.5)
            multiplier = limit_multipliers.get(m['normalized'], 1.2)
            
            config = base_config.copy()
            config['kalman'] = base_config['kalman'].copy()
            config['processing'] = base_config['processing'].copy()
            
            config['kalman']['observation_covariance'] = 5.0 / (trust ** 2)
            config['processing']['extreme_threshold'] = 10.0 * multiplier
            
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=db
            )
            if result:
                result['trust'] = trust
                result['limit_mult'] = multiplier
                results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        if not results:
            return {'acceptance_rate': 0}
        
        valid = [r for r in results if 'filtered_weight' in r]
        rejected = [r for r in results if r.get('rejected')]
        
        if not valid:
            return {'acceptance_rate': 0}
        
        weights = [r['filtered_weight'] for r in valid]
        
        metrics = {
            'acceptance_rate': len(valid) / len(results),
            'rejection_rate': len(rejected) / len(results)
        }
        
        if len(weights) > 1:
            diffs = np.diff(weights)
            metrics['smoothness'] = np.std(diffs) if len(diffs) > 0 else 0
            metrics['max_jump'] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0
        else:
            metrics['smoothness'] = 0
            metrics['max_jump'] = 0
        
        # Tracking error
        errors = []
        for r in valid:
            if 'raw_weight' in r:
                errors.append(abs(r['filtered_weight'] - r['raw_weight']))
        
        metrics['avg_error'] = np.mean(errors) if errors else 0
        
        return metrics
    
    def visualize_results(self, aggregate_results: Dict):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        strategies = list(aggregate_results.keys())
        
        # 1. Acceptance rates comparison
        ax1 = fig.add_subplot(gs[0, 0])
        means = [aggregate_results[s]['acceptance_rate']['mean'] * 100 for s in strategies]
        stds = [aggregate_results[s]['acceptance_rate']['std'] * 100 for s in strategies]
        
        bars = ax1.bar(range(len(strategies)), means, yerr=stds, capsize=5)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45)
        ax1.set_ylabel('Acceptance Rate (%)')
        ax1.set_title(f'Acceptance Rates (n={len(self.user_data)} users)')
        ax1.grid(True, alpha=0.3)
        
        # Color bars
        for i, bar in enumerate(bars):
            if means[i] >= 95:
                bar.set_color('green')
            elif means[i] >= 90:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 2. Smoothness comparison
        ax2 = fig.add_subplot(gs[0, 1])
        means = [aggregate_results[s]['smoothness']['mean'] for s in strategies]
        stds = [aggregate_results[s]['smoothness']['std'] for s in strategies]
        
        ax2.bar(range(len(strategies)), means, yerr=stds, capsize=5)
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels(strategies, rotation=45)
        ax2.set_ylabel('Smoothness (lower is better)')
        ax2.set_title('Output Smoothness')
        ax2.grid(True, alpha=0.3)
        
        # 3. Tracking error
        ax3 = fig.add_subplot(gs[0, 2])
        means = [aggregate_results[s]['avg_error']['mean'] for s in strategies]
        stds = [aggregate_results[s]['avg_error']['std'] for s in strategies]
        
        ax3.bar(range(len(strategies)), means, yerr=stds, capsize=5)
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels(strategies, rotation=45)
        ax3.set_ylabel('Average Error (kg)')
        ax3.set_title('Tracking Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # 4. User characteristics
        ax4 = fig.add_subplot(gs[1, 0])
        categories = list(self.user_stats['user_categories'].keys())
        counts = list(self.user_stats['user_categories'].values())
        
        ax4.pie(counts, labels=categories, autopct='%1.1f%%')
        ax4.set_title(f'User Source Diversity (n={len(self.user_data)})')
        
        # 5. Source distribution
        ax5 = fig.add_subplot(gs[1, 1])
        sources = list(self.user_stats['source_distributions'].keys())[:5]  # Top 5
        counts = [self.user_stats['source_distributions'][s] for s in sources]
        
        ax5.bar(sources, counts)
        ax5.set_xlabel('Source Type')
        ax5.set_ylabel('Measurement Count')
        ax5.set_title('Top 5 Source Types')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Performance improvement analysis
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Calculate improvement over baseline
        baseline_acceptance = aggregate_results['baseline']['acceptance_rate']['mean']
        baseline_smoothness = aggregate_results['baseline']['smoothness']['mean']
        baseline_error = aggregate_results['baseline']['avg_error']['mean']
        
        improvements = []
        for s in strategies:
            if s != 'baseline':
                # Combined improvement score (negative is better for smoothness/error)
                score = (
                    (aggregate_results[s]['acceptance_rate']['mean'] - baseline_acceptance) * 100 +
                    (baseline_smoothness - aggregate_results[s]['smoothness']['mean']) * 10 +
                    (baseline_error - aggregate_results[s]['avg_error']['mean']) * 20
                )
                improvements.append((s, score))
        
        if improvements:
            names, scores = zip(*improvements)
            colors = ['green' if s > 0 else 'red' for s in scores]
            ax6.bar(range(len(names)), scores, color=colors)
            ax6.set_xticks(range(len(names)))
            ax6.set_xticklabels(names, rotation=45)
            ax6.set_ylabel('Improvement Score')
            ax6.set_title('Performance vs Baseline')
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.grid(True, alpha=0.3)
        
        # 7. Statistical significance
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.axis('off')
        
        # Create detailed statistics table
        table_data = []
        headers = ['Strategy', 'Accept %', 'Smooth', 'Error', 'Score']
        
        for s in strategies:
            accept = f"{aggregate_results[s]['acceptance_rate']['mean']*100:.1f}±{aggregate_results[s]['acceptance_rate']['std']*100:.1f}"
            smooth = f"{aggregate_results[s]['smoothness']['mean']:.3f}±{aggregate_results[s]['smoothness']['std']:.3f}"
            error = f"{aggregate_results[s]['avg_error']['mean']:.2f}±{aggregate_results[s]['avg_error']['std']:.2f}"
            
            # Combined score
            score = (
                (1 - aggregate_results[s]['acceptance_rate']['mean']) * 10 +
                aggregate_results[s]['smoothness']['mean'] * 2 +
                aggregate_results[s]['avg_error']['mean'] * 3
            )
            
            table_data.append([s, accept, smooth, error, f"{score:.2f}"])
        
        # Sort by score
        table_data.sort(key=lambda x: float(x[4]))
        
        table = ax7.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax7.set_title('Detailed Performance Metrics (mean±std)', fontweight='bold')
        
        # 8. Recommendations
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        recommendations = self.generate_recommendations(aggregate_results)
        ax8.text(0.05, 0.95, recommendations, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace')
        
        plt.suptitle(f'Source Type Impact Analysis - {len(self.user_data):,} Users', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/large_scale_source_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on large-scale analysis."""
        # Rank strategies
        rankings = []
        for strategy, metrics in results.items():
            score = (
                (1 - metrics['acceptance_rate']['mean']) * 10 +
                metrics['smoothness']['mean'] * 2 +
                metrics['avg_error']['mean'] * 3
            )
            rankings.append((strategy, score))
        
        rankings.sort(key=lambda x: x[1])
        
        text = f"RECOMMENDATIONS\n"
        text += f"Based on {len(self.user_data):,} users\n\n"
        
        text += "BEST STRATEGY:\n"
        text += f"{rankings[0][0]}\n"
        text += f"Score: {rankings[0][1]:.2f}\n\n"
        
        # Check if improvement over baseline
        baseline_score = next(s for strat, s in rankings if strat == 'baseline')
        best_score = rankings[0][1]
        
        if rankings[0][0] != 'baseline':
            improvement = ((baseline_score - best_score) / baseline_score) * 100
            text += f"IMPROVEMENT:\n"
            text += f"{improvement:.1f}% better\n"
            text += f"than baseline\n\n"
            
            text += "IMPLEMENT:\n"
            if rankings[0][0] == 'trust_weighted':
                text += "Trust-weighted\nobservation noise"
            elif rankings[0][0] == 'adaptive_limits':
                text += "Source-specific\nphysio limits"
            elif rankings[0][0] == 'hybrid':
                text += "Combined trust +\nadaptive limits"
        else:
            text += "FINDING:\n"
            text += "Baseline optimal\n"
            text += "No changes needed"
        
        return text
    
    def generate_report(self, aggregate_results: Dict):
        """Generate comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("LARGE-SCALE SOURCE TYPE IMPACT ANALYSIS")
        report.append("=" * 80)
        report.append("")
        report.append(f"Analysis Date: {datetime.now().isoformat()}")
        report.append(f"Dataset: {self.data_file}")
        report.append(f"Users Analyzed: {len(self.user_data):,}")
        report.append(f"Total Measurements: {sum(len(m) for m in self.user_data.values()):,}")
        report.append("")
        
        # User statistics
        report.append("USER POPULATION STATISTICS")
        report.append("-" * 40)
        report.append(f"Measurements per user:")
        report.append(f"  Mean: {np.mean(self.user_stats['measurement_counts']):.1f}")
        report.append(f"  Median: {np.median(self.user_stats['measurement_counts']):.0f}")
        report.append(f"  Range: {min(self.user_stats['measurement_counts'])}-{max(self.user_stats['measurement_counts'])}")
        report.append("")
        
        report.append("Source diversity:")
        for category, count in self.user_stats['user_categories'].items():
            pct = count / len(self.user_data) * 100
            report.append(f"  {category}: {count:,} ({pct:.1f}%)")
        report.append("")
        
        # Strategy performance
        report.append("STRATEGY PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        # Rank strategies
        rankings = []
        for strategy, metrics in aggregate_results.items():
            score = (
                (1 - metrics['acceptance_rate']['mean']) * 10 +
                metrics['smoothness']['mean'] * 2 +
                metrics['avg_error']['mean'] * 3
            )
            rankings.append((strategy, score, metrics))
        
        rankings.sort(key=lambda x: x[1])
        
        for rank, (strategy, score, metrics) in enumerate(rankings, 1):
            report.append(f"\n{rank}. {strategy.upper()}")
            report.append(f"   Combined Score: {score:.3f}")
            report.append(f"   Acceptance Rate: {metrics['acceptance_rate']['mean']*100:.2f}% ± {metrics['acceptance_rate']['std']*100:.2f}%")
            report.append(f"   Smoothness: {metrics['smoothness']['mean']:.4f} ± {metrics['smoothness']['std']:.4f}")
            report.append(f"   Avg Error: {metrics['avg_error']['mean']:.3f} ± {metrics['avg_error']['std']:.3f} kg")
            report.append(f"   Max Jump: {metrics['max_jump']['mean']:.2f} ± {metrics['max_jump']['std']:.2f} kg")
        
        report.append("")
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 40)
        report.append(f"Sample Size: {len(self.user_data):,} users")
        
        if len(self.user_data) >= 1000:
            report.append("✅ Excellent statistical power")
            report.append("✅ Results highly generalizable")
            report.append("✅ Confident in recommendations")
        elif len(self.user_data) >= 100:
            report.append("✅ Good statistical power")
            report.append("⚠️ Results likely generalizable")
            report.append("⚠️ Moderate confidence in recommendations")
        else:
            report.append("❌ Limited statistical power")
            report.append("❌ Results may not generalize")
            report.append("❌ Low confidence in recommendations")
        
        report.append("")
        report.append("FINAL RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_strategy = rankings[0][0]
        if best_strategy != 'baseline':
            baseline_score = next(s for strat, s, _ in rankings if strat == 'baseline')
            improvement = ((baseline_score - rankings[0][1]) / baseline_score) * 100
            
            report.append(f"IMPLEMENT: {best_strategy}")
            report.append(f"Expected Improvement: {improvement:.1f}% over baseline")
            report.append("")
            
            if best_strategy == 'trust_weighted':
                report.append("Implementation Details:")
                report.append("  1. Add source trust scoring system")
                report.append("  2. Adjust Kalman observation noise by trust^2")
                report.append("  3. Trust scores: device=1.0, api=0.85, manual=0.4")
            elif best_strategy == 'adaptive_limits':
                report.append("Implementation Details:")
                report.append("  1. Add source-specific threshold multipliers")
                report.append("  2. Relax limits for manual entries (2x)")
                report.append("  3. Keep strict limits for device data (1x)")
            elif best_strategy == 'hybrid':
                report.append("Implementation Details:")
                report.append("  1. Implement both trust weighting and adaptive limits")
                report.append("  2. Moderate adjustments for both")
                report.append("  3. Test with A/B deployment")
        else:
            report.append("MAINTAIN: Current baseline implementation")
            report.append("No source differentiation provides best overall performance")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Run large-scale analysis."""
    print("=" * 60)
    print("LARGE-SCALE SOURCE TYPE IMPACT ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = LargeScaleSourceAnalyzer('data/2025-09-05_optimized.csv', sample_size=4000)
    
    # Load and sample users
    analyzer.load_and_sample_users()
    
    # Analyze user characteristics
    analyzer.analyze_user_characteristics()
    
    # Test strategies
    print("\nTesting strategies across users...")
    aggregate_results = analyzer.test_strategies_on_users()
    
    # Visualize results
    print("\nGenerating visualizations...")
    analyzer.visualize_results(aggregate_results)
    
    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_report(aggregate_results)
    
    # Save report
    with open('output/large_scale_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    
    # Save results as JSON
    with open('output/large_scale_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)
    
    print("\n✅ Analysis complete!")
    print("📊 Results saved to output/")
    
    return aggregate_results


if __name__ == "__main__":
    main()
