"""
Comprehensive analysis of different strategies for using source types.
Tests various approaches including resets, thresholds, and ensemble methods.
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.processor import WeightProcessor
from src.database import ProcessorStateDB
from src.visualization import normalize_source_type


class SourceStrategyAnalyzer:
    """Test different strategies for leveraging source type information."""
    
    def __init__(self, data_file: str):
        self.data = self.load_data(data_file)
        
    def load_data(self, file_path: str) -> Dict[str, List]:
        """Load and organize data by user."""
        user_data = defaultdict(list)
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['weight'] and row['effectiveDateTime']:
                    user_data[row['user_id']].append({
                        'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                        'weight': float(row['weight']),
                        'source': row['source_type'],
                        'normalized_source': normalize_source_type(row['source_type'])
                    })
        
        # Sort by timestamp
        for user_id in user_data:
            user_data[user_id].sort(key=lambda x: x['timestamp'])
            
        return dict(user_data)
    
    def get_config(self) -> Dict:
        """Get base configuration."""
        return {
            'processing': {
                'extreme_threshold': 10.0,
                'max_weight': 400.0,
                'min_weight': 30.0,
                'rate_limit_kg_per_day': 2.0,
                'outlier_threshold_kg': 5.0
            },
            'kalman': {
                'observation_covariance': 5.0,
                'transition_covariance_weight': 0.01,
                'transition_covariance_trend': 0.0001,
                'initial_variance': 10.0
            }
        }
    
    def strategy_baseline(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Baseline strategy - no source differentiation."""
        results = []
        db = ProcessorStateDB()
        config = self.get_config()
        
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
        
        return results, {'strategy': 'baseline', 'description': 'No source differentiation'}
    
    def strategy_reset_on_manual(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Reset state when manual/questionnaire entries appear."""
        results = []
        db = ProcessorStateDB()
        config = self.get_config()
        reset_count = 0
        
        for m in measurements:
            # Reset on manual entries
            if m['normalized_source'] in ['manual', 'questionnaire']:
                db.clear_state('test')
                reset_count += 1
            
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
                result['state_reset'] = m['normalized_source'] in ['manual', 'questionnaire']
                results.append(result)
        
        return results, {
            'strategy': 'reset_on_manual',
            'description': 'Reset state on manual/questionnaire entries',
            'reset_count': reset_count
        }
    
    def strategy_adaptive_thresholds(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Use different thresholds based on source reliability."""
        results = []
        db = ProcessorStateDB()
        base_config = self.get_config()
        
        # Source-specific threshold multipliers
        threshold_multipliers = {
            'patient-device': 1.0,     # Strict
            'api': 1.2,                # Slightly relaxed
            'questionnaire': 1.5,       # More relaxed
            'manual': 2.0,             # Most relaxed
            'other': 1.3
        }
        
        for m in measurements:
            # Adjust thresholds based on source
            config = base_config.copy()
            multiplier = threshold_multipliers.get(m['normalized_source'], 1.3)
            config['processing'] = base_config['processing'].copy()
            config['processing']['extreme_threshold'] = base_config['processing']['extreme_threshold'] * multiplier
            config['processing']['outlier_threshold_kg'] = base_config['processing']['outlier_threshold_kg'] * multiplier
            
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
                result['threshold_multiplier'] = multiplier
                results.append(result)
        
        return results, {
            'strategy': 'adaptive_thresholds',
            'description': 'Source-specific physiological thresholds'
        }
    
    def strategy_trust_weighted(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Weight Kalman filter by source trust."""
        results = []
        db = ProcessorStateDB()
        base_config = self.get_config()
        
        # Trust scores
        trust_scores = {
            'patient-device': 1.0,
            'api': 0.85,
            'questionnaire': 0.6,
            'manual': 0.4,
            'other': 0.5
        }
        
        for m in measurements:
            # Adjust observation noise based on trust
            trust = trust_scores.get(m['normalized_source'], 0.5)
            config = base_config.copy()
            config['kalman'] = base_config['kalman'].copy()
            # Lower trust = higher observation noise
            config['kalman']['observation_covariance'] = base_config['kalman']['observation_covariance'] / (trust ** 2)
            
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
                result['trust_score'] = trust
                result['obs_noise'] = config['kalman']['observation_covariance']
                results.append(result)
        
        return results, {
            'strategy': 'trust_weighted',
            'description': 'Trust-weighted Kalman observation noise'
        }
    
    def strategy_ensemble(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Maintain separate filters per source type and combine."""
        results = []
        
        # Separate databases for each source type
        dbs = {
            'patient-device': ProcessorStateDB(),
            'api': ProcessorStateDB(),
            'questionnaire': ProcessorStateDB(),
            'manual': ProcessorStateDB(),
            'other': ProcessorStateDB()
        }
        
        config = self.get_config()
        
        # Process through appropriate filter
        for m in measurements:
            source_type = m['normalized_source']
            if source_type not in dbs:
                source_type = 'other'
            
            # Process with source-specific filter
            result = WeightProcessor.process_weight(
                user_id='test',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=config['processing'],
                kalman_config=config['kalman'],
                db=dbs[source_type]
            )
            
            if result:
                # Also get predictions from other filters for ensemble
                ensemble_predictions = []
                for src, db in dbs.items():
                    if src != source_type:
                        state = db.get_state('test')
                        if state and state.get('last_state') is not None:
                            # Simple prediction: use last filtered value
                            ensemble_predictions.append(state['last_state'][0])
                
                # Combine predictions (weighted average)
                if ensemble_predictions:
                    weights = {'patient-device': 0.4, 'api': 0.3, 'questionnaire': 0.2, 'manual': 0.1}
                    weight_sum = weights.get(source_type, 0.2)
                    ensemble_value = result['filtered_weight if "filtered_weight" in result else result.get("raw_weight", m["weight"])'] * weight_sum
                    
                    for pred in ensemble_predictions:
                        ensemble_value += pred * (1 - weight_sum) / len(ensemble_predictions)
                    
                    result['ensemble_weight'] = ensemble_value
                else:
                    result['ensemble_weight'] = result['filtered_weight']
                
                results.append(result)
        
        return results, {
            'strategy': 'ensemble',
            'description': 'Ensemble of source-specific filters'
        }
    
    def strategy_hybrid(self, measurements: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Hybrid approach combining multiple strategies."""
        results = []
        db = ProcessorStateDB()
        base_config = self.get_config()
        reset_count = 0
        
        # Combined approach
        trust_scores = {
            'patient-device': 1.0,
            'api': 0.85,
            'questionnaire': 0.6,
            'manual': 0.4,
            'other': 0.5
        }
        
        threshold_multipliers = {
            'patient-device': 1.0,
            'api': 1.1,
            'questionnaire': 1.3,
            'manual': 1.5,
            'other': 1.2
        }
        
        for m in measurements:
            # Reset on care team uploads (questionnaire)
            if m['normalized_source'] == 'questionnaire':
                # Only reset if it's been a while since last measurement
                state = db.get_state('test')
                if state and state.get('last_timestamp'):
                    time_gap = m['timestamp'] - state['last_timestamp']
                    if time_gap.days > 7:  # Reset if gap > 7 days
                        db.clear_state('test')
                        reset_count += 1
            
            # Adjust both observation noise and thresholds
            trust = trust_scores.get(m['normalized_source'], 0.5)
            multiplier = threshold_multipliers.get(m['normalized_source'], 1.2)
            
            config = base_config.copy()
            config['kalman'] = base_config['kalman'].copy()
            config['processing'] = base_config['processing'].copy()
            
            config['kalman']['observation_covariance'] = base_config['kalman']['observation_covariance'] / (trust ** 2)
            config['processing']['extreme_threshold'] = base_config['processing']['extreme_threshold'] * multiplier
            
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
                result['trust_score'] = trust
                result['threshold_multiplier'] = multiplier
                results.append(result)
        
        return results, {
            'strategy': 'hybrid',
            'description': 'Combined trust weighting + adaptive thresholds + conditional resets',
            'reset_count': reset_count
        }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive metrics."""
        if not results:
            return {}
        
        valid = [r for r in results if r.get('filtered_weight')]
        rejected = [r for r in results if r.get('rejected')]
        
        if not valid:
            return {'acceptance_rate': 0.0}
        
        weights = [r['filtered_weight'] for r in valid]
        raw_weights = [r['raw_weight'] for r in valid if 'raw_weight' in r]
        
        metrics = {
            'acceptance_rate': len(valid) / len(results),
            'rejection_rate': len(rejected) / len(results),
            'total_processed': len(results),
            'total_accepted': len(valid)
        }
        
        if len(weights) > 1:
            # Smoothness
            diffs = np.diff(weights)
            metrics['smoothness'] = np.std(diffs)
            metrics['max_jump'] = np.max(np.abs(diffs))
            
            # Stability
            metrics['stability'] = np.std(weights)
            
            # Lag (how quickly we respond to changes)
            if raw_weights and len(raw_weights) > 1:
                raw_diffs = np.diff(raw_weights[:len(weights)])
                filtered_diffs = np.diff(weights)
                if len(raw_diffs) > 0:
                    # Correlation between raw and filtered changes
                    if np.std(raw_diffs) > 0 and np.std(filtered_diffs) > 0:
                        metrics['responsiveness'] = np.corrcoef(raw_diffs, filtered_diffs[:len(raw_diffs)])[0, 1]
                    else:
                        metrics['responsiveness'] = 0
        
        # Tracking error
        if raw_weights:
            errors = [abs(w - r) for w, r in zip(weights[:len(raw_weights)], raw_weights)]
            metrics['avg_error'] = np.mean(errors)
            metrics['max_error'] = np.max(errors)
        
        return metrics
    
    def run_analysis(self) -> Dict:
        """Run all strategies and compare."""
        # Find user with good data
        test_user = None
        for user_id, data in self.data.items():
            if len(data) >= 30:  # Need enough data
                sources = set(m['normalized_source'] for m in data)
                if len(sources) >= 2:  # Need source diversity
                    test_user = user_id
                    break
        
        if not test_user:
            print("No suitable user found")
            return {}
        
        measurements = self.data[test_user]
        print(f"Testing with user {test_user[:8]}... ({len(measurements)} measurements)")
        
        # Test all strategies
        strategies = [
            self.strategy_baseline,
            self.strategy_reset_on_manual,
            self.strategy_adaptive_thresholds,
            self.strategy_trust_weighted,
            self.strategy_ensemble,
            self.strategy_hybrid
        ]
        
        results = {}
        for strategy_func in strategies:
            print(f"Testing {strategy_func.__name__}...")
            strategy_results, info = strategy_func(measurements)
            metrics = self.calculate_metrics(strategy_results)
            results[info['strategy']] = {
                'results': strategy_results,
                'metrics': metrics,
                'info': info
            }
        
        return results
    
    def visualize_results(self, results: Dict):
        """Create comprehensive visualization."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        strategies = list(results.keys())
        
        # 1. Acceptance rates
        ax1 = fig.add_subplot(gs[0, 0])
        rates = [results[s]['metrics'].get('acceptance_rate', 0) * 100 for s in strategies]
        bars = ax1.bar(range(len(strategies)), rates)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45)
        ax1.set_ylabel('Acceptance Rate (%)')
        ax1.set_title('Acceptance Rates by Strategy')
        ax1.grid(True, alpha=0.3)
        
        # Color code bars
        for i, bar in enumerate(bars):
            if rates[i] >= 95:
                bar.set_color('green')
            elif rates[i] >= 90:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 2. Smoothness comparison
        ax2 = fig.add_subplot(gs[0, 1])
        smoothness = [results[s]['metrics'].get('smoothness', 0) for s in strategies]
        ax2.bar(range(len(strategies)), smoothness)
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels(strategies, rotation=45)
        ax2.set_ylabel('Smoothness (lower is better)')
        ax2.set_title('Output Smoothness')
        ax2.grid(True, alpha=0.3)
        
        # 3. Tracking error
        ax3 = fig.add_subplot(gs[0, 2])
        errors = [results[s]['metrics'].get('avg_error', 0) for s in strategies]
        ax3.bar(range(len(strategies)), errors)
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels(strategies, rotation=45)
        ax3.set_ylabel('Average Error (kg)')
        ax3.set_title('Tracking Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # 4. Best strategy time series
        ax4 = fig.add_subplot(gs[1, :])
        
        # Find best strategy by combined score
        best_strategy = min(strategies,
                           key=lambda s: (1 - results[s]['metrics'].get('acceptance_rate', 0)) * 10 +
                                        results[s]['metrics'].get('smoothness', 0) * 2 +
                                        results[s]['metrics'].get('avg_error', 0) * 3)
        
        best_results = results[best_strategy]['results']
        if best_results:
            valid = [r for r in best_results if r.get('filtered_weight')]
            if valid:
                timestamps = [r['timestamp'] for r in valid]
                raw = [r['raw_weight'] for r in valid]
                filtered = [r['filtered_weight'] for r in valid]
                
                ax4.plot(timestamps, raw, 'o', alpha=0.3, markersize=4, label='Raw')
                ax4.plot(timestamps, filtered, '-', linewidth=2, label='Filtered')
                
                # Show ensemble if available
                if 'ensemble_weight' in valid[0]:
                    ensemble = [r.get('ensemble_weight', r['filtered_weight']) for r in valid]
                    ax4.plot(timestamps, ensemble, '--', alpha=0.7, label='Ensemble')
                
                ax4.set_title(f'Best Strategy: {best_strategy}')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Weight (kg)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Responsiveness vs Smoothness tradeoff
        ax5 = fig.add_subplot(gs[2, 0])
        for s in strategies:
            smooth = results[s]['metrics'].get('smoothness', 0)
            resp = results[s]['metrics'].get('responsiveness', 0)
            ax5.scatter(smooth, resp, s=100)
            ax5.annotate(s, (smooth, resp), fontsize=8)
        
        ax5.set_xlabel('Smoothness (lower is better)')
        ax5.set_ylabel('Responsiveness (higher is better)')
        ax5.set_title('Smoothness vs Responsiveness')
        ax5.grid(True, alpha=0.3)
        
        # 6. Strategy comparison radar chart
        ax6 = fig.add_subplot(gs[2, 1], projection='polar')
        
        # Metrics for radar
        metrics = ['Acceptance', 'Smoothness\n(inverted)', 'Low Error', 'Stability\n(inverted)']
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot top 3 strategies
        top_strategies = sorted(strategies,
                               key=lambda s: (1 - results[s]['metrics'].get('acceptance_rate', 0)) * 10 +
                                            results[s]['metrics'].get('smoothness', 0) * 2 +
                                            results[s]['metrics'].get('avg_error', 0) * 3)[:3]
        
        for strategy in top_strategies:
            m = results[strategy]['metrics']
            values = [
                m.get('acceptance_rate', 0),
                1 / (1 + m.get('smoothness', 0)),  # Invert so higher is better
                1 / (1 + m.get('avg_error', 0)),   # Invert so higher is better
                1 / (1 + m.get('stability', 0))    # Invert so higher is better
            ]
            values += values[:1]  # Complete the circle
            
            ax6.plot(angles, values, 'o-', linewidth=2, label=strategy)
            ax6.fill(angles, values, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_ylim(0, 1)
        ax6.set_title('Strategy Performance Profile')
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
        
        # 7. Summary and recommendations
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        summary = self.generate_summary(results)
        ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace')
        
        plt.suptitle('Source Type Strategy Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/source_strategies_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_summary(self, results: Dict) -> str:
        """Generate summary and recommendations."""
        # Rank strategies
        rankings = []
        for strategy, data in results.items():
            m = data['metrics']
            score = (
                (1 - m.get('acceptance_rate', 0)) * 10 +
                m.get('smoothness', 0) * 2 +
                m.get('avg_error', 0) * 3
            )
            rankings.append((strategy, score, m))
        
        rankings.sort(key=lambda x: x[1])
        
        text = "STRATEGY RANKINGS\n\n"
        for i, (strategy, score, metrics) in enumerate(rankings[:3]):
            text += f"{i+1}. {strategy}\n"
            text += f"   Score: {score:.2f}\n"
            text += f"   Accept: {metrics.get('acceptance_rate', 0)*100:.1f}%\n"
            text += f"   Smooth: {metrics.get('smoothness', 0):.3f}\n"
            text += f"   Error: {metrics.get('avg_error', 0):.3f} kg\n\n"
        
        # Insights
        text += "KEY INSIGHTS:\n\n"
        
        # Check if trust weighting helps
        if 'trust_weighted' in results and 'baseline' in results:
            trust_score = rankings[[r[0] for r in rankings].index('trust_weighted')]
            base_score = rankings[[r[0] for r in rankings].index('baseline')]
            if trust_score < base_score:
                text += "✓ Trust weighting improves\n  performance\n\n"
        
        # Check if hybrid is best
        if rankings[0][0] == 'hybrid':
            text += "✓ Combined approach works\n  best\n\n"
        
        # Check reset impact
        if 'reset_on_manual' in results:
            reset_info = results['reset_on_manual']['info']
            if reset_info.get('reset_count', 0) > 0:
                text += f"! {reset_info['reset_count']} resets triggered\n\n"
        
        text += "RECOMMENDATION:\n"
        if rankings[0][0] == 'baseline':
            text += "Current approach is optimal\nfor this dataset"
        else:
            text += f"Implement {rankings[0][0]}\nstrategy for better\nperformance"
        
        return text


def main():
    """Run strategy analysis."""
    print("Analyzing source type strategies...")
    
    analyzer = SourceStrategyAnalyzer('data/test_sample.csv')
    results = analyzer.run_analysis()
    
    if results:
        analyzer.visualize_results(results)
        
        # Generate detailed report
        report = []
        report.append("=" * 60)
        report.append("SOURCE TYPE STRATEGY ANALYSIS")
        report.append("=" * 60)
        
        for strategy, data in results.items():
            report.append(f"\n{strategy.upper()}")
            report.append("-" * 30)
            report.append(data['info']['description'])
            report.append("\nMetrics:")
            for metric, value in data['metrics'].items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
        
        report_text = "\n".join(report)
        
        with open('output/source_strategies_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print("\nAnalysis complete! Check output/ for results.")
    
    return results


if __name__ == "__main__":
    main()
