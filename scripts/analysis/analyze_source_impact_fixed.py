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
from src.processor_database import ProcessorStateDB, get_state_db
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
    
    def get_base_config(self) -> Dict:
        """Get base configuration for processing."""
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
    
    def simulate_trust_weighted_kalman(
        self, 
        measurements: List[Dict], 
        trust_model: Dict[str, float]
    ) -> List[Dict]:
        """Simulate processing with trust-weighted Kalman observation noise."""
        results = []
        db = ProcessorStateDB()
        base_config = self.get_base_config()
        
        for m in measurements:
            source_trust = trust_model.get(m['normalized_source'], 0.5)
            
            # Adjust observation noise inversely to trust
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
    
    def simulate_baseline(self, measurements: List[Dict]) -> List[Dict]:
        """Simulate baseline processing without source differentiation."""
        results = []
        db = ProcessorStateDB()
        base_config = self.get_base_config()
        
        for m in measurements:
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
        
        # Calculate tracking accuracy
        tracking_errors = []
        for r in valid_results:
            if 'raw_weight' in r:
                error = abs(r['filtered_weight'] - r['raw_weight'])
                tracking_errors.append(error)
        
        avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
        
        # Calculate stability
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
    
    def run_comparison(self) -> Dict:
        """Run simplified comparison of trust models."""
        # Select test user
        test_user_id = self.select_diverse_user()
        if not test_user_id:
            print("No suitable user found for testing")
            return {}
        
        measurements = self.data[test_user_id]
        print(f"Testing with user {test_user_id[:8]}... ({len(measurements)} measurements)")
        
        comparison_results = {}
        
        # Test baseline
        print("Testing baseline...")
        baseline_results = self.simulate_baseline(measurements)
        comparison_results['baseline'] = {
            'results': baseline_results,
            'metrics': self.calculate_metrics(baseline_results)
        }
        
        # Test each trust model
        for model_name, trust_model in self.TRUST_MODELS.items():
            if model_name != 'baseline':
                print(f"Testing {model_name}...")
                results = self.simulate_trust_weighted_kalman(measurements, trust_model)
                comparison_results[model_name] = {
                    'results': results,
                    'metrics': self.calculate_metrics(results)
                }
        
        return comparison_results
    
    def visualize_comparison(self, results: Dict):
        """Create visualization comparing different trust models."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Source Trust Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison bar chart
        ax = axes[0, 0]
        models = list(results.keys())
        metrics_names = ['acceptance_rate', 'smoothness', 'avg_tracking_error']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics_names):
            values = [results[m]['metrics'].get(metric, 0) for m in models]
            if metric == 'acceptance_rate':
                values = [v * 100 for v in values]  # Convert to percentage
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Trust Model')
        ax.set_ylabel('Value')
        ax.set_title('Key Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Time series comparison for best model
        ax = axes[0, 1]
        
        # Find best model by combined score
        best_model = min(results.keys(), 
                        key=lambda m: (1 - results[m]['metrics'].get('acceptance_rate', 0)) * 5 +
                                     results[m]['metrics'].get('smoothness', 0) * 2 +
                                     results[m]['metrics'].get('avg_tracking_error', 0) * 3)
        
        best_results = results[best_model]['results']
        if best_results:
            timestamps = [r['timestamp'] for r in best_results if r.get('filtered_weight')]
            raw = [r['raw_weight'] for r in best_results if r.get('filtered_weight')]
            filtered = [r['filtered_weight'] for r in best_results if r.get('filtered_weight')]
            
            ax.plot(timestamps, raw, 'o', alpha=0.3, label='Raw', markersize=4)
            ax.plot(timestamps, filtered, '-', linewidth=2, label='Filtered')
            ax.set_title(f'Best Model: {best_model}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Weight (kg)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Acceptance rates
        ax = axes[1, 0]
        acceptance_rates = [results[m]['metrics'].get('acceptance_rate', 0) * 100 for m in models]
        bars = ax.bar(models, acceptance_rates)
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Acceptance Rates by Model')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if acceptance_rates[i] > 90:
                bar.set_color('green')
            elif acceptance_rates[i] > 80:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 4. Smoothness comparison
        ax = axes[1, 1]
        smoothness = [results[m]['metrics'].get('smoothness', 0) for m in models]
        bars = ax.bar(models, smoothness)
        ax.set_ylabel('Smoothness (lower is better)')
        ax.set_title('Output Smoothness')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 5. Tracking error
        ax = axes[2, 0]
        errors = [results[m]['metrics'].get('avg_tracking_error', 0) for m in models]
        bars = ax.bar(models, errors)
        ax.set_ylabel('Average Tracking Error (kg)')
        ax.set_title('Tracking Accuracy')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 6. Summary recommendations
        ax = axes[2, 1]
        ax.axis('off')
        
        recommendations = self.generate_recommendations(results)
        ax.text(0.1, 0.9, recommendations, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('output/source_trust_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on analysis."""
        # Find best performers
        metrics_summary = []
        for model, data in results.items():
            metrics = data['metrics']
            score = (
                (1 - metrics.get('acceptance_rate', 0)) * 5 +
                metrics.get('smoothness', 0) * 2 +
                metrics.get('avg_tracking_error', 0) * 3
            )
            metrics_summary.append((model, score, metrics))
        
        metrics_summary.sort(key=lambda x: x[1])
        
        text = "RECOMMENDATIONS\n\n"
        text += "Best Performing Models:\n"
        for i, (model, score, metrics) in enumerate(metrics_summary[:3]):
            text += f"{i+1}. {model}\n"
            text += f"   Score: {score:.2f}\n"
            text += f"   Accept: {metrics.get('acceptance_rate', 0)*100:.1f}%\n"
            text += f"   Smooth: {metrics.get('smoothness', 0):.3f}\n"
            text += f"   Error: {metrics.get('avg_tracking_error', 0):.3f} kg\n\n"
        
        # Specific recommendations
        if metrics_summary[0][0] != 'baseline':
            text += "Key Insight:\n"
            text += "Source-aware processing improves\n"
            text += "quality without losing data.\n\n"
            text += "Implementation Priority:\n"
            text += "1. Trust-weighted observation noise\n"
            text += "2. Source-specific thresholds\n"
            text += "3. Conditional resets for uploads"
        else:
            text += "Current baseline performs well.\n"
            text += "Source differentiation may not\n"
            text += "be necessary for this dataset."
        
        return text
    
    def generate_report(self, results: Dict) -> str:
        """Generate detailed text report."""
        report = []
        report.append("=" * 60)
        report.append("SOURCE TYPE IMPACT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        
        for model, data in results.items():
            metrics = data['metrics']
            report.append(f"\n{model.upper()}:")
            report.append(f"  Acceptance Rate: {metrics.get('acceptance_rate', 0)*100:.1f}%")
            report.append(f"  Smoothness: {metrics.get('smoothness', 0):.4f}")
            report.append(f"  Avg Tracking Error: {metrics.get('avg_tracking_error', 0):.3f} kg")
            report.append(f"  Stability: {metrics.get('stability', 0):.4f}")
            report.append(f"  Max Jump: {metrics.get('max_jump', 0):.2f} kg")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        report.append(self.generate_recommendations(results))
        
        return "\n".join(report)


def main():
    """Run the source impact analysis."""
    print("Starting source type impact analysis...")
    
    # Initialize analyzer
    analyzer = SourceTrustAnalyzer('data/test_sample.csv')
    
    # Run comparison
    print("Running simulations...")
    results = analyzer.run_comparison()
    
    if results:
        # Generate visualizations
        print("Creating visualizations...")
        analyzer.visualize_comparison(results)
        
        # Generate report
        print("Generating report...")
        report = analyzer.generate_report(results)
        
        # Save report
        with open('output/source_trust_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + report)
        print("\nAnalysis complete! Results saved to output/")
    else:
        print("Analysis failed - no suitable data found")
    
    return results


if __name__ == "__main__":
    main()
