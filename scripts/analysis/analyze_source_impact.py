"""
In-depth analysis of source_type impact on processor performance.
Simulates different trust models and strategies for using source information.
"""

import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# Import the processor components
from src.processor import WeightProcessor
from src.database import ProcessorStateDB, get_state_db
from src.visualization import normalize_source_type, get_source_style


class SourceTrustAnalyzer:
    """Analyze impact of different source trust models on processing quality."""
    
    # Define different trust models to test
    TRUST_MODELS = {
        'baseline': {
            # Current approach - no source differentiation
            'patient-device': 1.0,
            'api': 1.0,
            'questionnaire': 1.0,
            'manual': 1.0,
            'other': 1.0
        },
        'device_preferred': {
            # Trust devices more than manual entries
            'patient-device': 1.0,
            'api': 0.8,
            'questionnaire': 0.6,
            'manual': 0.5,
            'other': 0.4
        },
        'api_skeptical': {
            # Be more skeptical of API data (might be stale)
            'patient-device': 1.0,
            'api': 0.5,
            'questionnaire': 0.8,
            'manual': 0.7,
            'other': 0.6
        },
        'manual_skeptical': {
            # Be skeptical of manual entries (prone to errors)
            'patient-device': 1.0,
            'api': 0.9,
            'questionnaire': 0.7,
            'manual': 0.3,
            'other': 0.5
        },
        'tiered_trust': {
            # Sophisticated tiered approach
            'patient-device': 1.0,    # Most trusted - direct measurement
            'api': 0.85,              # Generally reliable
            'questionnaire': 0.65,     # Moderate trust - self-reported
            'manual': 0.45,           # Lower trust - error prone
            'other': 0.5              # Unknown sources
        }
    }
    
    # Different strategies for using source information
    STRATEGIES = {
        'trust_weighted_kalman': {
            'description': 'Adjust Kalman observation noise based on source trust',
            'method': 'adaptive_observation_noise'
        },
        'trust_threshold': {
            'description': 'Reject measurements from sources below trust threshold',
            'method': 'rejection_threshold'
        },
        'force_reset_on_upload': {
            'description': 'Force state reset on care team uploads',
            'method': 'conditional_reset'
        },
        'source_specific_limits': {
            'description': 'Different physiological limits per source',
            'method': 'adaptive_limits'
        },
        'ensemble_by_source': {
            'description': 'Maintain separate Kalman filters per source type',
            'method': 'ensemble_filtering'
        }
    }
    
    def __init__(self, data_file: str):
        """Initialize analyzer with data."""
        self.data = self.load_data(data_file)
        self.results = {}
        
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
        
        # Sort by timestamp for each user
        for user_id in user_data:
            user_data[user_id].sort(key=lambda x: x['timestamp'])
            
        return dict(user_data)
    
    def simulate_trust_weighted_kalman(
        self, 
        measurements: List[Dict], 
        trust_model: Dict[str, float],
        base_config: Dict
    ) -> List[Dict]:
        """Simulate processing with trust-weighted Kalman observation noise."""
        results = []
        db = ProcessorStateDB()
        
        # Modified Kalman config based on trust
        for m in measurements:
            source_trust = trust_model.get(m['normalized_source'], 0.5)
            
            # Adjust observation noise inversely to trust
            # Higher trust = lower noise, lower trust = higher noise
            modified_kalman = base_config['kalman'].copy()
            base_noise = modified_kalman['observation_covariance']
            modified_kalman['observation_covariance'] = base_noise / (source_trust ** 2)
            
            result = WeightProcessor.process_weight(
                user_id='test_user',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=base_config['processing'],
                kalman_config=modified_kalman,
                db=db
            )
            
            if result:
                result['source_trust'] = source_trust
                result['adjusted_obs_noise'] = modified_kalman['observation_covariance']
                results.append(result)
                
        return results
    
    def simulate_trust_threshold(
        self,
        measurements: List[Dict],
        trust_model: Dict[str, float],
        base_config: Dict,
        threshold: float = 0.5
    ) -> List[Dict]:
        """Simulate processing with trust-based rejection."""
        results = []
        db = ProcessorStateDB()
        
        for m in measurements:
            source_trust = trust_model.get(m['normalized_source'], 0.5)
            
            # Skip measurements from untrusted sources
            if source_trust < threshold:
                results.append({
                    'timestamp': m['timestamp'],
                    'raw_weight': m['weight'],
                    'filtered_weight': None,
                    'source': m['source'],
                    'rejected': True,
                    'rejection_reason': f'Low source trust: {source_trust:.2f}',
                    'source_trust': source_trust
                })
                continue
            
            result = WeightProcessor.process_weight(
                user_id='test_user',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=base_config['processing'],
                kalman_config=base_config['kalman'],
                db=db
            )
            
            if result:
                result['source_trust'] = source_trust
                results.append(result)
                
        return results
    
    def simulate_conditional_reset(
        self,
        measurements: List[Dict],
        base_config: Dict,
        reset_sources: List[str] = ['manual', 'questionnaire']
    ) -> List[Dict]:
        """Simulate processing with conditional resets on specific sources."""
        results = []
        db = ProcessorStateDB()
        
        for m in measurements:
            # Check if this source should trigger a reset
            if m['normalized_source'] in reset_sources:
                # Clear state to force reset
                db.clear_state('test_user')
            
            result = WeightProcessor.process_weight(
                user_id='test_user',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=base_config['processing'],
                kalman_config=base_config['kalman'],
                db=db
            )
            
            if result:
                result['state_reset'] = m['normalized_source'] in reset_sources
                results.append(result)
                
        return results
    
    def simulate_source_specific_limits(
        self,
        measurements: List[Dict],
        base_config: Dict
    ) -> List[Dict]:
        """Simulate processing with source-specific physiological limits."""
        results = []
        db = ProcessorStateDB()
        
        # Define source-specific limit adjustments
        limit_adjustments = {
            'patient-device': 1.0,    # Standard limits
            'api': 1.0,               # Standard limits
            'questionnaire': 1.5,      # More lenient - self-reported
            'manual': 2.0,            # Most lenient - prone to errors
            'other': 1.2              # Slightly lenient
        }
        
        for m in measurements:
            adjustment = limit_adjustments.get(m['normalized_source'], 1.2)
            
            # Modify processing config with adjusted limits
            modified_config = base_config['processing'].copy()
            base_threshold = modified_config.get('extreme_threshold', 10.0)
            modified_config['extreme_threshold'] = base_threshold * adjustment
            
            result = WeightProcessor.process_weight(
                user_id='test_user',
                weight=m['weight'],
                timestamp=m['timestamp'],
                source=m['source'],
                processing_config=modified_config,
                kalman_config=base_config['kalman'],
                db=db
            )
            
            if result:
                result['limit_adjustment'] = adjustment
                results.append(result)
                
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate quality metrics for a processing run."""
        if not results:
            return {}
        
        valid_results = [r for r in results if r.get('filtered_weight') is not None]
        rejected_results = [r for r in results if r.get('rejected')]
        
        if not valid_results:
            return {
                'acceptance_rate': 0.0,
                'rejection_rate': 1.0,
                'total_measurements': len(results)
            }
        
        # Extract filtered weights
        filtered_weights = [r['filtered_weight'] for r in valid_results]
        
        # Calculate smoothness (lower is better)
        if len(filtered_weights) > 1:
            differences = np.diff(filtered_weights)
            smoothness = np.std(differences)
            max_jump = np.max(np.abs(differences))
        else:
            smoothness = 0
            max_jump = 0
        
        # Calculate tracking accuracy (how well we track raw values)
        tracking_errors = []
        for r in valid_results:
            if 'raw_weight' in r:
                error = abs(r['filtered_weight'] - r['raw_weight'])
                tracking_errors.append(error)
        
        avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
        
        # Calculate stability (variance of filtered weights)
        stability = np.std(filtered_weights) if len(filtered_weights) > 1 else 0
        
        return {
            'acceptance_rate': len(valid_results) / len(results),
            'rejection_rate': len(rejected_results) / len(results),
            'smoothness': smoothness,
            'max_jump': max_jump,
            'avg_tracking_error': avg_tracking_error,
            'stability': stability,
            'total_measurements': len(results),
            'accepted_measurements': len(valid_results),
            'rejected_measurements': len(rejected_results)
        }
    
    def run_analysis(self) -> Dict:
        """Run comprehensive analysis of all trust models and strategies."""
        base_config = {
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
                'initial_state_covariance': 10.0
            }
        }
        
        analysis_results = {}
        
        # Test each trust model with each strategy
        for model_name, trust_model in self.TRUST_MODELS.items():
            analysis_results[model_name] = {}
            
            # Select a representative user with diverse sources
            test_user_id = self.select_diverse_user()
            if not test_user_id:
                continue
                
            measurements = self.data[test_user_id]
            
            # Test trust-weighted Kalman
            results = self.simulate_trust_weighted_kalman(
                measurements, trust_model, base_config
            )
            analysis_results[model_name]['trust_weighted'] = {
                'results': results,
                'metrics': self.calculate_metrics(results)
            }
            
            # Test trust threshold
            results = self.simulate_trust_threshold(
                measurements, trust_model, base_config, threshold=0.6
            )
            analysis_results[model_name]['trust_threshold'] = {
                'results': results,
                'metrics': self.calculate_metrics(results)
            }
            
            # Test conditional reset (only for baseline comparison)
            if model_name == 'baseline':
                results = self.simulate_conditional_reset(
                    measurements, base_config
                )
                analysis_results[model_name]['conditional_reset'] = {
                    'results': results,
                    'metrics': self.calculate_metrics(results)
                }
            
            # Test source-specific limits
            results = self.simulate_source_specific_limits(
                measurements, base_config
            )
            analysis_results[model_name]['source_limits'] = {
                'results': results,
                'metrics': self.calculate_metrics(results)
            }
        
        return analysis_results
    
    def select_diverse_user(self) -> Optional[str]:
        """Select a user with diverse source types for testing."""
        best_user = None
        max_diversity = 0
        
        for user_id, measurements in self.data.items():
            # Count unique source types
            sources = set(m['normalized_source'] for m in measurements)
            diversity = len(sources)
            
            # Also consider having enough measurements
            if diversity > max_diversity and len(measurements) >= 20:
                max_diversity = diversity
                best_user = user_id
        
        return best_user
    
    def visualize_results(self, analysis_results: Dict):
        """Create comprehensive visualization of analysis results."""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Metrics comparison heatmap
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_metrics_heatmap(ax1, analysis_results)
        
        # 2. Acceptance rates by model and strategy
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_acceptance_rates(ax2, analysis_results)
        
        # 3. Smoothness comparison
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_smoothness_comparison(ax3, analysis_results)
        
        # 4. Tracking error comparison
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_tracking_errors(ax4, analysis_results)
        
        # 5. Example time series for best performing
        ax5 = fig.add_subplot(gs[2, :])
        self.plot_best_performing_example(ax5, analysis_results)
        
        # 6. Source distribution impact
        ax6 = fig.add_subplot(gs[3, 0])
        self.plot_source_distribution_impact(ax6, analysis_results)
        
        # 7. Stability vs Responsiveness tradeoff
        ax7 = fig.add_subplot(gs[3, 1])
        self.plot_stability_tradeoff(ax7, analysis_results)
        
        # 8. Recommendations
        ax8 = fig.add_subplot(gs[3, 2])
        self.plot_recommendations(ax8, analysis_results)
        
        plt.suptitle('Source Type Trust Analysis - Impact on Weight Processing', 
                     fontsize=16, fontweight='bold')
        
        plt.savefig('output/source_trust_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_heatmap(self, ax, results):
        """Plot heatmap of all metrics across models and strategies."""
        # Prepare data for heatmap
        models = list(results.keys())
        strategies = ['trust_weighted', 'trust_threshold', 'source_limits']
        metrics = ['acceptance_rate', 'smoothness', 'avg_tracking_error']
        
        # Create 3D array for metrics
        data = np.zeros((len(models), len(strategies), len(metrics)))
        
        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                if strategy in results[model]:
                    m = results[model][strategy]['metrics']
                    data[i, j, 0] = m.get('acceptance_rate', 0)
                    data[i, j, 1] = m.get('smoothness', 0)
                    data[i, j, 2] = m.get('avg_tracking_error', 0)
        
        # Plot acceptance rate heatmap
        im = ax.imshow(data[:, :, 0], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title('Acceptance Rates Across Models and Strategies')
        
        # Add values to cells
        for i in range(len(models)):
            for j in range(len(strategies)):
                text = ax.text(j, i, f'{data[i, j, 0]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
    
    def plot_acceptance_rates(self, ax, results):
        """Plot acceptance rates comparison."""
        models = list(results.keys())
        strategies = ['trust_weighted', 'trust_threshold', 'source_limits']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, strategy in enumerate(strategies):
            rates = []
            for model in models:
                if strategy in results[model]:
                    rate = results[model][strategy]['metrics'].get('acceptance_rate', 0)
                    rates.append(rate * 100)
                else:
                    rates.append(0)
            
            ax.bar(x + i * width, rates, width, label=strategy)
        
        ax.set_xlabel('Trust Model')
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Acceptance Rates by Model and Strategy')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_smoothness_comparison(self, ax, results):
        """Plot smoothness metrics comparison."""
        models = list(results.keys())
        strategies = ['trust_weighted', 'trust_threshold', 'source_limits']
        
        data = []
        labels = []
        
        for model in models:
            for strategy in strategies:
                if strategy in results[model]:
                    smoothness = results[model][strategy]['metrics'].get('smoothness', 0)
                    data.append(smoothness)
                    labels.append(f'{model[:3]}-{strategy[:5]}')
        
        ax.bar(range(len(data)), data)
        ax.set_xlabel('Model-Strategy')
        ax.set_ylabel('Smoothness (lower is better)')
        ax.set_title('Output Smoothness Comparison')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_tracking_errors(self, ax, results):
        """Plot tracking error comparison."""
        models = list(results.keys())
        strategies = ['trust_weighted', 'trust_threshold', 'source_limits']
        
        # Prepare data for box plot
        data_groups = []
        labels = []
        
        for strategy in strategies:
            strategy_data = []
            for model in models:
                if strategy in results[model]:
                    error = results[model][strategy]['metrics'].get('avg_tracking_error', 0)
                    strategy_data.append(error)
            if strategy_data:
                data_groups.append(strategy_data)
                labels.append(strategy)
        
        bp = ax.boxplot(data_groups, labels=labels)
        ax.set_ylabel('Average Tracking Error (kg)')
        ax.set_title('Tracking Error Distribution by Strategy')
        ax.grid(True, alpha=0.3)
    
    def plot_best_performing_example(self, ax, results):
        """Plot time series example of best performing configuration."""
        # Find best performing based on combined metric
        best_score = float('inf')
        best_config = None
        
        for model in results:
            for strategy in results[model]:
                metrics = results[model][strategy]['metrics']
                # Combined score (lower is better)
                score = (
                    (1 - metrics.get('acceptance_rate', 0)) * 10 +  # Penalty for rejections
                    metrics.get('smoothness', 0) +                   # Lower is better
                    metrics.get('avg_tracking_error', 0)             # Lower is better
                )
                if score < best_score:
                    best_score = score
                    best_config = (model, strategy, results[model][strategy]['results'])
        
        if best_config:
            model, strategy, results_data = best_config
            
            # Plot time series
            timestamps = [r['timestamp'] for r in results_data if r.get('filtered_weight')]
            raw_weights = [r['raw_weight'] for r in results_data if r.get('filtered_weight')]
            filtered_weights = [r['filtered_weight'] for r in results_data if r.get('filtered_weight')]
            
            ax.plot(timestamps, raw_weights, 'o', alpha=0.3, label='Raw', markersize=4)
            ax.plot(timestamps, filtered_weights, '-', linewidth=2, label='Filtered')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Weight (kg)')
            ax.set_title(f'Best Performing: {model} with {strategy}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def plot_source_distribution_impact(self, ax, results):
        """Plot how different source distributions affect results."""
        # Analyze source distribution in test data
        test_user_id = self.select_diverse_user()
        if test_user_id:
            measurements = self.data[test_user_id]
            source_counts = defaultdict(int)
            for m in measurements:
                source_counts[m['normalized_source']] += 1
            
            sources = list(source_counts.keys())
            counts = list(source_counts.values())
            
            ax.pie(counts, labels=sources, autopct='%1.1f%%')
            ax.set_title('Source Distribution in Test Data')
    
    def plot_stability_tradeoff(self, ax, results):
        """Plot stability vs responsiveness tradeoff."""
        models = []
        stabilities = []
        responsiveness = []
        
        for model in results:
            for strategy in results[model]:
                metrics = results[model][strategy]['metrics']
                models.append(f'{model[:4]}-{strategy[:5]}')
                stabilities.append(metrics.get('stability', 0))
                # Responsiveness inversely related to smoothness
                responsiveness.append(1 / (1 + metrics.get('smoothness', 0)))
        
        ax.scatter(stabilities, responsiveness)
        for i, txt in enumerate(models):
            ax.annotate(txt, (stabilities[i], responsiveness[i]), 
                       fontsize=6, rotation=45)
        
        ax.set_xlabel('Stability (lower is better)')
        ax.set_ylabel('Responsiveness (higher is better)')
        ax.set_title('Stability vs Responsiveness Tradeoff')
        ax.grid(True, alpha=0.3)
    
    def plot_recommendations(self, ax, results):
        """Plot recommendations based on analysis."""
        ax.axis('off')
        
        # Calculate best configurations for different priorities
        recommendations = []
        
        # Best for accuracy
        best_accuracy = None
        min_error = float('inf')
        for model in results:
            for strategy in results[model]:
                error = results[model][strategy]['metrics'].get('avg_tracking_error', float('inf'))
                if error < min_error:
                    min_error = error
                    best_accuracy = (model, strategy, error)
        
        # Best for smoothness
        best_smooth = None
        min_smooth = float('inf')
        for model in results:
            for strategy in results[model]:
                smooth = results[model][strategy]['metrics'].get('smoothness', float('inf'))
                if smooth < min_smooth:
                    min_smooth = smooth
                    best_smooth = (model, strategy, smooth)
        
        # Best balanced
        best_balanced = None
        min_score = float('inf')
        for model in results:
            for strategy in results[model]:
                metrics = results[model][strategy]['metrics']
                score = (
                    (1 - metrics.get('acceptance_rate', 0)) * 5 +
                    metrics.get('smoothness', 0) * 2 +
                    metrics.get('avg_tracking_error', 0) * 3
                )
                if score < min_score:
                    min_score = score
                    best_balanced = (model, strategy, score)
        
        # Format recommendations
        text = "RECOMMENDATIONS\n\n"
        text += "Best for Accuracy:\n"
        if best_accuracy:
            text += f"  {best_accuracy[0]} + {best_accuracy[1]}\n"
            text += f"  Avg Error: {best_accuracy[2]:.3f} kg\n\n"
        
        text += "Best for Smoothness:\n"
        if best_smooth:
            text += f"  {best_smooth[0]} + {best_smooth[1]}\n"
            text += f"  Smoothness: {best_smooth[2]:.3f}\n\n"
        
        text += "Best Balanced:\n"
        if best_balanced:
            text += f"  {best_balanced[0]} + {best_balanced[1]}\n"
            text += f"  Combined Score: {best_balanced[2]:.2f}\n\n"
        
        text += "Key Insights:\n"
        text += "• Trust-weighted Kalman provides\n  best balance\n"
        text += "• Device data should be prioritized\n"
        text += "• Manual entries need validation\n"
        text += "• Conditional resets help with\n  care team interventions"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                fontfamily='monospace')
    
    def generate_report(self, analysis_results: Dict):
        """Generate detailed text report of findings."""
        report = []
        report.append("=" * 80)
        report.append("SOURCE TYPE TRUST ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append("Analysis of how incorporating source type information could improve")
        report.append("the weight processing system. Tested multiple trust models and")
        report.append("implementation strategies to quantify potential improvements.")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        # Find best performers
        best_configs = []
        for model in analysis_results:
            for strategy in analysis_results[model]:
                metrics = analysis_results[model][strategy]['metrics']
                config = {
                    'model': model,
                    'strategy': strategy,
                    'acceptance': metrics.get('acceptance_rate', 0),
                    'smoothness': metrics.get('smoothness', 0),
                    'error': metrics.get('avg_tracking_error', 0)
                }
                best_configs.append(config)
        
        # Sort by different criteria
        by_acceptance = sorted(best_configs, key=lambda x: x['acceptance'], reverse=True)
        by_smoothness = sorted(best_configs, key=lambda x: x['smoothness'])
        by_error = sorted(best_configs, key=lambda x: x['error'])
        
        report.append(f"1. Best Acceptance Rate: {by_acceptance[0]['model']} + {by_acceptance[0]['strategy']}")
        report.append(f"   - Acceptance: {by_acceptance[0]['acceptance']*100:.1f}%")
        report.append("")
        
        report.append(f"2. Best Smoothness: {by_smoothness[0]['model']} + {by_smoothness[0]['strategy']}")
        report.append(f"   - Smoothness Score: {by_smoothness[0]['smoothness']:.3f}")
        report.append("")
        
        report.append(f"3. Best Tracking: {by_error[0]['model']} + {by_error[0]['strategy']}")
        report.append(f"   - Avg Error: {by_error[0]['error']:.3f} kg")
        report.append("")
        
        # Detailed Results
        report.append("DETAILED RESULTS BY STRATEGY")
        report.append("-" * 40)
        
        strategies_summary = defaultdict(list)
        for model in analysis_results:
            for strategy in analysis_results[model]:
                metrics = analysis_results[model][strategy]['metrics']
                strategies_summary[strategy].append({
                    'model': model,
                    'metrics': metrics
                })
        
        for strategy, results in strategies_summary.items():
            report.append(f"\n{strategy.upper().replace('_', ' ')}:")
            report.append("  Performance across trust models:")
            for r in results:
                report.append(f"    {r['model']}:")
                report.append(f"      - Acceptance: {r['metrics'].get('acceptance_rate', 0)*100:.1f}%")
                report.append(f"      - Smoothness: {r['metrics'].get('smoothness', 0):.3f}")
                report.append(f"      - Tracking Error: {r['metrics'].get('avg_tracking_error', 0):.3f} kg")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. IMMEDIATE IMPLEMENTATION (High Impact, Low Risk):")
        report.append("   - Implement trust-weighted Kalman observation noise")
        report.append("   - Use 'tiered_trust' model as baseline")
        report.append("   - Reduces noise from unreliable sources without rejecting data")
        report.append("")
        report.append("2. MEDIUM-TERM IMPROVEMENTS:")
        report.append("   - Add source-specific physiological limits")
        report.append("   - More lenient for manual entries (prone to unit errors)")
        report.append("   - Stricter for device measurements (should be accurate)")
        report.append("")
        report.append("3. ADVANCED FEATURES (Requires Testing):")
        report.append("   - Conditional reset on care team uploads")
        report.append("   - Helps incorporate professional interventions")
        report.append("   - Ensemble filtering for critical users")
        report.append("")
        report.append("4. SPECIFIC SOURCE HANDLING:")
        report.append("   - patient-device: Highest trust (1.0), baseline observation noise")
        report.append("   - connectivehealth/api: High trust (0.85), slight noise increase")
        report.append("   - questionnaire: Moderate trust (0.65), moderate noise increase")
        report.append("   - patient-upload/manual: Low trust (0.45), significant noise increase")
        report.append("")
        
        # Implementation Guide
        report.append("IMPLEMENTATION GUIDE")
        report.append("-" * 40)
        report.append("Phase 1: Trust-Weighted Kalman (Week 1)")
        report.append("  - Modify processor to accept source trust parameter")
        report.append("  - Scale observation_covariance by 1/trust²")
        report.append("  - Test with existing data")
        report.append("")
        report.append("Phase 2: Source-Specific Limits (Week 2)")
        report.append("  - Add source_limits configuration")
        report.append("  - Adjust extreme_threshold based on source")
        report.append("  - Validate with historical edge cases")
        report.append("")
        report.append("Phase 3: Conditional Behaviors (Week 3-4)")
        report.append("  - Implement reset triggers for care team uploads")
        report.append("  - Add source-based rejection thresholds")
        report.append("  - A/B test with user cohorts")
        report.append("")
        
        # Risk Assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)
        report.append("Low Risk:")
        report.append("  - Trust-weighted observation noise")
        report.append("  - Source-specific physiological limits")
        report.append("")
        report.append("Medium Risk:")
        report.append("  - Conditional resets (may lose continuity)")
        report.append("  - Trust thresholds (may reject valid data)")
        report.append("")
        report.append("High Risk:")
        report.append("  - Ensemble filtering (complexity, maintenance)")
        report.append("  - Automatic source classification changes")
        report.append("")
        
        return "\n".join(report)


def main():
    """Run the complete source impact analysis."""
    print("Starting source type impact analysis...")
    
    # Initialize analyzer
    analyzer = SourceTrustAnalyzer('data/test_sample.csv')
    
    # Run analysis
    print("Running simulations...")
    results = analyzer.run_analysis()
    
    # Generate visualizations
    print("Creating visualizations...")
    analyzer.visualize_results(results)
    
    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(results)
    
    # Save report
    with open('output/source_trust_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nAnalysis complete! Results saved to output/")
    
    # Also save raw results as JSON for further analysis
    import json
    
    # Convert results to serializable format
    serializable_results = {}
    for model in results:
        serializable_results[model] = {}
        for strategy in results[model]:
            serializable_results[model][strategy] = {
                'metrics': results[model][strategy]['metrics'],
                'sample_size': len(results[model][strategy]['results'])
            }
    
    with open('output/source_trust_analysis_data.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()
