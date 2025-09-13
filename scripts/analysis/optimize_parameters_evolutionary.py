"""
Evolutionary algorithm to find optimal parameters for quality gates and Kalman filter.
Tests across multiple users to find globally optimal configuration.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random
import sys
import os
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.processor import WeightProcessor
from src.database import get_state_db

@dataclass
class Parameters:
    """Parameter set for optimization."""
    # Physiological limits
    max_change_1h_absolute: float
    max_change_6h_absolute: float
    max_change_24h_absolute: float
    max_sustained_daily: float
    limit_tolerance: float
    sustained_tolerance: float
    session_variance_threshold: float
    
    # Kalman parameters
    initial_variance: float
    transition_covariance_weight: float
    transition_covariance_trend: float
    observation_covariance: float
    
    # Fitness tracking
    fitness: float = 0.0
    acceptance_rate: float = 0.0
    avg_std_dev: float = 0.0
    avg_smoothness: float = 0.0
    
    def to_configs(self) -> Tuple[Dict, Dict]:
        """Convert to processing and kalman configs."""
        processing_config = {
            'min_weight': 20.0,
            'max_weight': 300.0,
            'extreme_threshold': 10.0,
            'max_daily_change': 0.05,
            'physiological': {
                'enable_physiological_limits': True,
                'max_change_1h_percent': 0.02,
                'max_change_1h_absolute': self.max_change_1h_absolute,
                'max_change_6h_percent': 0.025,
                'max_change_6h_absolute': self.max_change_6h_absolute,
                'max_change_24h_percent': 0.035,
                'max_change_24h_absolute': self.max_change_24h_absolute,
                'max_sustained_daily': self.max_sustained_daily,
                'limit_tolerance': self.limit_tolerance,
                'sustained_tolerance': self.sustained_tolerance,
                'session_timeout_minutes': 5,
                'session_variance_threshold': self.session_variance_threshold
            }
        }
        
        kalman_config = {
            'initial_variance': self.initial_variance,
            'transition_covariance_weight': self.transition_covariance_weight,
            'transition_covariance_trend': self.transition_covariance_trend,
            'observation_covariance': self.observation_covariance
        }
        
        return processing_config, kalman_config
    
    def mutate(self, mutation_rate: float = 0.2, mutation_strength: float = 0.1):
        """Mutate parameters with given probability and strength."""
        if random.random() < mutation_rate:
            self.max_change_1h_absolute *= (1 + random.gauss(0, mutation_strength))
            self.max_change_1h_absolute = np.clip(self.max_change_1h_absolute, 2.0, 6.0)
        
        if random.random() < mutation_rate:
            self.max_change_6h_absolute *= (1 + random.gauss(0, mutation_strength))
            self.max_change_6h_absolute = np.clip(self.max_change_6h_absolute, 3.0, 8.0)
        
        if random.random() < mutation_rate:
            self.max_change_24h_absolute *= (1 + random.gauss(0, mutation_strength))
            self.max_change_24h_absolute = np.clip(self.max_change_24h_absolute, 4.0, 10.0)
        
        if random.random() < mutation_rate:
            self.max_sustained_daily *= (1 + random.gauss(0, mutation_strength))
            self.max_sustained_daily = np.clip(self.max_sustained_daily, 1.0, 3.0)
        
        if random.random() < mutation_rate:
            self.limit_tolerance *= (1 + random.gauss(0, mutation_strength))
            self.limit_tolerance = np.clip(self.limit_tolerance, 0.05, 0.3)
        
        if random.random() < mutation_rate:
            self.sustained_tolerance *= (1 + random.gauss(0, mutation_strength))
            self.sustained_tolerance = np.clip(self.sustained_tolerance, 0.1, 0.5)
        
        if random.random() < mutation_rate:
            self.session_variance_threshold *= (1 + random.gauss(0, mutation_strength))
            self.session_variance_threshold = np.clip(self.session_variance_threshold, 3.0, 10.0)
        
        # Kalman mutations
        if random.random() < mutation_rate:
            self.initial_variance *= (1 + random.gauss(0, mutation_strength))
            self.initial_variance = np.clip(self.initial_variance, 0.1, 2.0)
        
        if random.random() < mutation_rate:
            self.transition_covariance_weight *= (1 + random.gauss(0, mutation_strength))
            self.transition_covariance_weight = np.clip(self.transition_covariance_weight, 0.01, 0.2)
        
        if random.random() < mutation_rate:
            self.transition_covariance_trend *= (1 + random.gauss(0, mutation_strength))
            self.transition_covariance_trend = np.clip(self.transition_covariance_trend, 0.0001, 0.01)
        
        if random.random() < mutation_rate:
            self.observation_covariance *= (1 + random.gauss(0, mutation_strength))
            self.observation_covariance = np.clip(self.observation_covariance, 0.5, 5.0)
    
    @staticmethod
    def crossover(parent1: 'Parameters', parent2: 'Parameters') -> 'Parameters':
        """Create offspring from two parents."""
        child = Parameters(
            # Randomly inherit from parents
            max_change_1h_absolute=random.choice([parent1.max_change_1h_absolute, parent2.max_change_1h_absolute]),
            max_change_6h_absolute=random.choice([parent1.max_change_6h_absolute, parent2.max_change_6h_absolute]),
            max_change_24h_absolute=random.choice([parent1.max_change_24h_absolute, parent2.max_change_24h_absolute]),
            max_sustained_daily=random.choice([parent1.max_sustained_daily, parent2.max_sustained_daily]),
            limit_tolerance=random.choice([parent1.limit_tolerance, parent2.limit_tolerance]),
            sustained_tolerance=random.choice([parent1.sustained_tolerance, parent2.sustained_tolerance]),
            session_variance_threshold=random.choice([parent1.session_variance_threshold, parent2.session_variance_threshold]),
            initial_variance=random.choice([parent1.initial_variance, parent2.initial_variance]),
            transition_covariance_weight=random.choice([parent1.transition_covariance_weight, parent2.transition_covariance_weight]),
            transition_covariance_trend=random.choice([parent1.transition_covariance_trend, parent2.transition_covariance_trend]),
            observation_covariance=random.choice([parent1.observation_covariance, parent2.observation_covariance])
        )
        return child
    
    @staticmethod
    def random():
        """Generate random parameters within reasonable bounds."""
        return Parameters(
            max_change_1h_absolute=random.uniform(2.5, 5.0),
            max_change_6h_absolute=random.uniform(3.5, 7.0),
            max_change_24h_absolute=random.uniform(4.5, 9.0),
            max_sustained_daily=random.uniform(1.2, 2.5),
            limit_tolerance=random.uniform(0.08, 0.25),
            sustained_tolerance=random.uniform(0.15, 0.40),
            session_variance_threshold=random.uniform(4.0, 8.0),
            initial_variance=random.uniform(0.3, 1.5),
            transition_covariance_weight=random.uniform(0.02, 0.15),
            transition_covariance_trend=random.uniform(0.0001, 0.005),
            observation_covariance=random.uniform(0.8, 3.0)
        )
    
    @staticmethod
    def baseline():
        """Current baseline parameters."""
        return Parameters(
            max_change_1h_absolute=3.0,
            max_change_6h_absolute=4.0,
            max_change_24h_absolute=5.0,
            max_sustained_daily=1.5,
            limit_tolerance=0.10,
            sustained_tolerance=0.25,
            session_variance_threshold=5.0,
            initial_variance=1.0,
            transition_covariance_weight=0.1,
            transition_covariance_trend=0.001,
            observation_covariance=1.0
        )

def calculate_smoothness(weights: List[float]) -> float:
    """Calculate smoothness metric (lower is better)."""
    if len(weights) < 3:
        return 0
    first_diff = np.diff(weights)
    second_diff = np.diff(first_diff)
    return np.std(second_diff)

def evaluate_parameters_for_user(
    params: Parameters,
    user_data: pd.DataFrame,
    user_id: str
) -> Dict[str, float]:
    """Evaluate parameters for a single user."""
    processing_config, kalman_config = params.to_configs()
    
    db = get_state_db()
    db.clear_state(user_id)
    
    accepted = 0
    total = 0
    filtered_weights = []
    weight_changes = []
    
    for _, row in user_data.iterrows():
        timestamp = datetime.fromisoformat(row['effectiveDateTime'])
        weight = row['weight']
        source = row['source_type']
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        total += 1
        if result and result.get('accepted'):
            accepted += 1
            filtered_weights.append(result.get('filtered_weight', weight))
            if len(filtered_weights) > 1:
                change = abs(filtered_weights[-1] - filtered_weights[-2])
                weight_changes.append(change)
    
    acceptance_rate = accepted / total if total > 0 else 0
    std_dev = np.std(filtered_weights) if len(filtered_weights) > 1 else 0
    smoothness = calculate_smoothness(filtered_weights) if len(filtered_weights) > 2 else 0
    avg_change = np.mean(weight_changes) if weight_changes else 0
    
    return {
        'acceptance_rate': acceptance_rate,
        'std_dev': std_dev,
        'smoothness': smoothness,
        'avg_change': avg_change,
        'accepted': accepted,
        'total': total
    }

def evaluate_parameters(params: Parameters, all_user_data: Dict[str, pd.DataFrame]) -> float:
    """Evaluate parameters across all users and calculate fitness."""
    total_accepted = 0
    total_measurements = 0
    all_acceptance_rates = []
    all_std_devs = []
    all_smoothness = []
    
    for user_id, user_data in all_user_data.items():
        metrics = evaluate_parameters_for_user(params, user_data, user_id)
        
        total_accepted += metrics['accepted']
        total_measurements += metrics['total']
        all_acceptance_rates.append(metrics['acceptance_rate'])
        if metrics['std_dev'] > 0:
            all_std_devs.append(metrics['std_dev'])
        if metrics['smoothness'] > 0:
            all_smoothness.append(metrics['smoothness'])
    
    # Calculate overall metrics
    overall_acceptance = total_accepted / total_measurements if total_measurements > 0 else 0
    avg_std_dev = np.mean(all_std_devs) if all_std_devs else 0
    avg_smoothness = np.mean(all_smoothness) if all_smoothness else 0
    
    # Fitness function (maximize acceptance, minimize std_dev and smoothness)
    # Weights: acceptance (0.5), stability (0.25), smoothness (0.25)
    fitness = (
        0.5 * overall_acceptance +  # Higher is better
        0.25 * (1 / (1 + avg_std_dev)) +  # Lower is better
        0.25 * (1 / (1 + avg_smoothness))  # Lower is better
    )
    
    # Penalize extreme parameters
    penalty = 0
    if params.max_change_24h_absolute > 8.0:  # Too loose
        penalty += 0.05
    if params.transition_covariance_trend > 0.002:  # Too flexible
        penalty += 0.05
    if params.observation_covariance > 3.5:  # Too much noise
        penalty += 0.05
    
    fitness -= penalty
    
    # Store metrics in params
    params.fitness = fitness
    params.acceptance_rate = overall_acceptance
    params.avg_std_dev = avg_std_dev
    params.avg_smoothness = avg_smoothness
    
    return fitness

class EvolutionaryOptimizer:
    """Evolutionary algorithm for parameter optimization."""
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.15
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.history = []
    
    def optimize(self, all_user_data: Dict[str, pd.DataFrame]) -> Parameters:
        """Run evolutionary optimization."""
        print(f"Starting evolutionary optimization...")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Users: {len(all_user_data)}, Total measurements: {sum(len(df) for df in all_user_data.values())}")
        print("=" * 80)
        
        # Initialize population
        population = []
        
        # Add baseline configuration
        baseline = Parameters.baseline()
        evaluate_parameters(baseline, all_user_data)
        population.append(baseline)
        print(f"Baseline fitness: {baseline.fitness:.4f} (Accept: {baseline.acceptance_rate:.1%})")
        
        # Add random individuals
        for i in range(self.population_size - 1):
            individual = Parameters.random()
            evaluate_parameters(individual, all_user_data)
            population.append(individual)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        print(f"Initial best fitness: {population[0].fitness:.4f}")
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Select elite
            elite = population[:self.elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)
                
                # Crossover
                child = Parameters.crossover(parent1, parent2)
                
                # Mutation
                child.mutate(self.mutation_rate, self.mutation_strength)
                
                # Evaluate
                evaluate_parameters(child, all_user_data)
                new_population.append(child)
            
            # Update population
            population = new_population
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track history
            best = population[0]
            self.history.append({
                'generation': generation + 1,
                'best_fitness': best.fitness,
                'avg_fitness': np.mean([p.fitness for p in population]),
                'best_acceptance': best.acceptance_rate,
                'best_std_dev': best.avg_std_dev,
                'best_smoothness': best.avg_smoothness
            })
            
            print(f"  Best fitness: {best.fitness:.4f}")
            print(f"  Acceptance: {best.acceptance_rate:.1%}, StdDev: {best.avg_std_dev:.3f}, Smooth: {best.avg_smoothness:.4f}")
            
            # Adaptive mutation
            if generation > 10:
                # Reduce mutation as we converge
                self.mutation_strength *= 0.98
        
        return population[0]
    
    def tournament_select(self, population: List[Parameters], tournament_size: int = 3) -> Parameters:
        """Tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def plot_history(self):
        """Plot optimization history."""
        if not self.history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        generations = [h['generation'] for h in self.history]
        
        # Fitness over time
        ax = axes[0, 0]
        ax.plot(generations, [h['best_fitness'] for h in self.history], 'b-', label='Best', linewidth=2)
        ax.plot(generations, [h['avg_fitness'] for h in self.history], 'r--', label='Average', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Acceptance rate
        ax = axes[0, 1]
        ax.plot(generations, [h['best_acceptance'] * 100 for h in self.history], 'g-', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Best Acceptance Rate')
        ax.grid(True, alpha=0.3)
        
        # Standard deviation
        ax = axes[1, 0]
        ax.plot(generations, [h['best_std_dev'] for h in self.history], 'orange', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Avg StdDev (kg)')
        ax.set_title('Weight Stability')
        ax.grid(True, alpha=0.3)
        
        # Smoothness
        ax = axes[1, 1]
        ax.plot(generations, [h['best_smoothness'] for h in self.history], 'purple', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Smoothness')
        ax.set_title('Trajectory Smoothness (Lower is Better)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_output/evolution_history.png', dpi=150, bbox_inches='tight')
        print("\nEvolution history saved to test_output/evolution_history.png")

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/2025-09-05_optimized.csv')
    
    # Convert weight to kg if needed
    df['weight'] = df.apply(lambda row: row['weight'] / 1000 if row['unit'] == 'g' else row['weight'], axis=1)
    
    # Users to optimize for
    user_ids = [
        "0040872d-333a-4ace-8c5a-b2fcd056e65a",
        "b1c7ec66-85f9-4ecc-b7b8-46742f5e78db",
        "42f31300-fae5-4719-a4e4-f63d61e624cc",
        "8823af48-caa8-4b57-9e2c-dc19c509f2e3",
        "1e87d3ab-20b1-479d-ad4d-8986e1af38da",
        "0093a653-476b-4401-bbec-33a89abc2b18",
        "4d5054c4-2492-4cd8-a15c-53381fb6bd49",
        "1a452430-7351-4b8c-b921-4fb17f8a29cc",
        "5a4e195f-47d4-4f6e-aab4-1e29a1830e63",
        "3e3ca1f5-3a99-4bc4-93e0-80781d7749e2",
        "77c8fbf5-2dca-419a-a602-93f47e3d5d84",
        "48f3ae0e-8688-4b15-8f30-537224646426"
    ]
    
    # Prepare user data
    all_user_data = {}
    for user_id in user_ids:
        user_data = df[df['user_id'] == user_id].sort_values('effectiveDateTime')
        if not user_data.empty:
            all_user_data[user_id] = user_data
            print(f"User {user_id[:8]}: {len(user_data)} measurements")
    
    print(f"\nTotal users for optimization: {len(all_user_data)}")
    
    # Run optimization
    optimizer = EvolutionaryOptimizer(
        population_size=40,
        generations=25,
        elite_size=8,
        mutation_rate=0.3,
        mutation_strength=0.15
    )
    
    best_params = optimizer.optimize(all_user_data)
    
    # Plot history
    optimizer.plot_history()
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print("\nBest Parameters Found:")
    print("-" * 40)
    print("Physiological Limits:")
    print(f"  max_change_1h_absolute: {best_params.max_change_1h_absolute:.2f} kg")
    print(f"  max_change_6h_absolute: {best_params.max_change_6h_absolute:.2f} kg")
    print(f"  max_change_24h_absolute: {best_params.max_change_24h_absolute:.2f} kg")
    print(f"  max_sustained_daily: {best_params.max_sustained_daily:.2f} kg/day")
    print(f"  limit_tolerance: {best_params.limit_tolerance:.2%}")
    print(f"  sustained_tolerance: {best_params.sustained_tolerance:.2%}")
    print(f"  session_variance_threshold: {best_params.session_variance_threshold:.2f} kg")
    
    print("\nKalman Filter:")
    print(f"  initial_variance: {best_params.initial_variance:.3f}")
    print(f"  transition_covariance_weight: {best_params.transition_covariance_weight:.4f}")
    print(f"  transition_covariance_trend: {best_params.transition_covariance_trend:.6f}")
    print(f"  observation_covariance: {best_params.observation_covariance:.3f}")
    
    print("\nPerformance Metrics:")
    print(f"  Fitness Score: {best_params.fitness:.4f}")
    print(f"  Acceptance Rate: {best_params.acceptance_rate:.1%}")
    print(f"  Avg StdDev: {best_params.avg_std_dev:.3f} kg")
    print(f"  Avg Smoothness: {best_params.avg_smoothness:.4f}")
    
    # Compare with baseline
    baseline = Parameters.baseline()
    evaluate_parameters(baseline, all_user_data)
    
    print("\nImprovement over Baseline:")
    print(f"  Fitness: {best_params.fitness:.4f} vs {baseline.fitness:.4f} ({(best_params.fitness - baseline.fitness) / baseline.fitness * 100:+.1f}%)")
    print(f"  Acceptance: {best_params.acceptance_rate:.1%} vs {baseline.acceptance_rate:.1%} ({(best_params.acceptance_rate - baseline.acceptance_rate) * 100:+.1f}pp)")
    print(f"  StdDev: {best_params.avg_std_dev:.3f} vs {baseline.avg_std_dev:.3f} ({(best_params.avg_std_dev - baseline.avg_std_dev) / baseline.avg_std_dev * 100:+.1f}%)")
    print(f"  Smoothness: {best_params.avg_smoothness:.4f} vs {baseline.avg_smoothness:.4f} ({(best_params.avg_smoothness - baseline.avg_smoothness) / baseline.avg_smoothness * 100:+.1f}%)")
    
    # Save results
    results = {
        'best_parameters': asdict(best_params),
        'baseline_parameters': asdict(baseline),
        'optimization_history': optimizer.history,
        'user_count': len(all_user_data),
        'total_measurements': sum(len(df) for df in all_user_data.values())
    }
    
    with open('test_output/optimal_parameters.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to test_output/optimal_parameters.json")

if __name__ == "__main__":
    main()
