"""
Evolutionary algorithm to optimize adaptive noise multipliers.
Uses actual data to find the best multipliers for each source.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import random
from collections import defaultdict

# Import our processor components
from src.processor import WeightProcessor
from src.database import get_state_db
from src.models import KALMAN_DEFAULTS, PROCESSING_DEFAULTS
from src.kalman import KalmanFilterManager

class AdaptiveMultiplierOptimizer:
    """Optimize adaptive noise multipliers using evolutionary algorithm."""
    
    def __init__(self, data_file: str, sample_size: int = 1000):
        """Initialize optimizer with data."""
        self.data_file = data_file
        self.sample_size = sample_size
        self.load_data()
        
    def load_data(self):
        """Load and prepare data for optimization."""
        print(f"Loading data from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        
        # Rename columns for consistency
        df['timestamp'] = pd.to_datetime(df['effectiveDateTime'])
        df['source'] = df['source_type']
        
        # Get unique sources
        self.sources = df['source'].unique().tolist()
        print(f"Found {len(self.sources)} unique sources")
        
        # Sample users with diverse source usage
        user_source_counts = df.groupby('user_id')['source'].nunique()
        diverse_users = user_source_counts[user_source_counts > 1].index.tolist()
        
        if len(diverse_users) > 50:
            sample_users = random.sample(diverse_users, min(50, len(diverse_users)))
        else:
            sample_users = diverse_users[:50]
        
        # Get data for sampled users
        self.test_data = df[df['user_id'].isin(sample_users)].copy()
        self.test_data = self.test_data.sort_values(['user_id', 'timestamp'])
        
        print(f"Using {len(sample_users)} users with {len(self.test_data)} measurements")
        
        # Group by user for faster access
        self.user_data = {}
        for user_id, user_df in self.test_data.groupby('user_id'):
            self.user_data[user_id] = user_df.to_dict('records')
    
    def evaluate_multipliers(self, multipliers: Dict[str, float]) -> float:
        """
        Evaluate a set of multipliers on the test data.
        Returns a fitness score (lower is better).
        """
        total_error = 0
        total_stability = 0
        total_confidence = 0
        measurement_count = 0
        
        # Process each user
        for user_id, measurements in self.user_data.items():
            if len(measurements) < 5:
                continue
            
            # Clear state for this user
            db = get_state_db()
            db.clear_state(user_id)
            
            # Process measurements with adaptive multipliers
            filtered_weights = []
            confidences = []
            
            for i, meas in enumerate(measurements):
                # Apply multiplier for this source
                kalman_config = KALMAN_DEFAULTS.copy()
                source_multiplier = multipliers.get(meas['source'], 1.0)
                kalman_config['observation_covariance'] *= source_multiplier
                
                # Process measurement
                result = WeightProcessor.process_weight(
                    user_id=user_id,
                    weight=meas['weight'],
                    timestamp=meas['timestamp'],
                    source=meas['source'],
                    kalman_config=kalman_config,
                    processing_config=PROCESSING_DEFAULTS.copy()
                )
                
                if result and result.get('accepted'):
                    filtered_weights.append(result['filtered_weight'])
                    confidences.append(result.get('confidence', 0))
            
            if len(filtered_weights) < 3:
                continue
            
            # Calculate metrics
            # 1. Prediction error (how well we predict next measurement)
            errors = []
            for i in range(len(filtered_weights) - 1):
                error = abs(filtered_weights[i] - measurements[i + 1]['weight'])
                errors.append(error)
            
            if errors:
                avg_error = np.mean(errors)
                total_error += avg_error
            
            # 2. Stability (smoothness of filtered weights)
            if len(filtered_weights) > 1:
                stability = np.std(np.diff(filtered_weights))
                total_stability += stability
            
            # 3. Average confidence
            if confidences:
                avg_confidence = np.mean(confidences)
                total_confidence += avg_confidence
            
            measurement_count += 1
        
        if measurement_count == 0:
            return float('inf')
        
        # Combine metrics into fitness score
        # Lower error is better, lower instability is better, higher confidence is better
        fitness = (
            (total_error / measurement_count) * 1.0 +  # Prediction error weight
            (total_stability / measurement_count) * 0.5 +  # Stability weight
            (1.0 - total_confidence / measurement_count) * 0.3  # Confidence weight (inverted)
        )
        
        return fitness
    
    def create_individual(self) -> Dict[str, float]:
        """Create a random individual (set of multipliers)."""
        individual = {}
        for source in self.sources:
            # Random multiplier between 0.3 and 5.0
            individual[source] = random.uniform(0.3, 5.0)
        return individual
    
    def mutate(self, individual: Dict[str, float], mutation_rate: float = 0.2) -> Dict[str, float]:
        """Mutate an individual."""
        mutated = individual.copy()
        for source in self.sources:
            if random.random() < mutation_rate:
                # Add gaussian noise
                change = np.random.normal(0, 0.5)
                mutated[source] = max(0.3, min(5.0, mutated[source] + change))
        return mutated
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Create offspring from two parents."""
        child = {}
        for source in self.sources:
            # Random weighted average
            weight = random.random()
            child[source] = weight * parent1[source] + (1 - weight) * parent2[source]
        return child
    
    def optimize(self, generations: int = 20, population_size: int = 20):
        """Run evolutionary optimization."""
        print("\n" + "=" * 80)
        print("EVOLUTIONARY OPTIMIZATION")
        print("=" * 80)
        
        # Initialize population
        print(f"\nInitializing population of {population_size} individuals...")
        population = []
        
        # Add current multipliers as one individual
        current_multipliers = {
            'care-team-upload': 0.5,
            'patient-upload': 0.7,
            'internal-questionnaire': 0.8,
            'patient-device': 1.0,
            'https://connectivehealth.io': 1.5,
            'https://api.iglucose.com': 3.0
        }
        
        # Fill in missing sources
        for source in self.sources:
            if source not in current_multipliers:
                current_multipliers[source] = 1.0
        
        population.append(current_multipliers)
        
        # Add random individuals
        for _ in range(population_size - 1):
            population.append(self.create_individual())
        
        # Evolution loop
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = self.evaluate_multipliers(individual)
                fitness_scores.append((fitness, individual))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                
                print(f"  Individual {i+1}: fitness = {fitness:.4f}")
            
            fitness_history.append(best_fitness)
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[0])
            
            # Select top performers
            elite_size = population_size // 4
            new_population = [ind for _, ind in fitness_scores[:elite_size]]
            
            # Create offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = random.choice([ind for _, ind in fitness_scores[:population_size//2]])
                parent2 = random.choice([ind for _, ind in fitness_scores[:population_size//2]])
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            print(f"  Best fitness so far: {best_fitness:.4f}")
        
        return best_individual, best_fitness, fitness_history

def main():
    # Initialize optimizer
    optimizer = AdaptiveMultiplierOptimizer("./data/2025-09-05_optimized.csv")
    
    # Run optimization
    best_multipliers, best_fitness, history = optimizer.optimize(
        generations=10,  # Reduced for faster testing
        population_size=10
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nBest fitness achieved: {best_fitness:.4f}")
    
    print("\nOptimized multipliers:")
    for source, multiplier in sorted(best_multipliers.items(), key=lambda x: x[1]):
        print(f"  {source}: {multiplier:.2f}")
    
    # Compare with current values
    current = {
        'care-team-upload': 0.5,
        'patient-upload': 0.7,
        'internal-questionnaire': 0.8,
        'patient-device': 1.0,
        'https://connectivehealth.io': 1.5,
        'https://api.iglucose.com': 3.0
    }
    
    print("\nComparison with current values:")
    for source in best_multipliers:
        old_val = current.get(source, 1.0)
        new_val = best_multipliers[source]
        change = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
        print(f"  {source}: {old_val:.2f} → {new_val:.2f} ({change:+.1f}%)")
    
    # Save results
    results = {
        'optimized_multipliers': best_multipliers,
        'best_fitness': best_fitness,
        'fitness_history': history,
        'optimization_date': datetime.now().isoformat()
    }
    
    with open('test_output/optimal_adaptive_multipliers.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_output/optimal_adaptive_multipliers.json")
    
    # Generate config snippet
    print("\n" + "=" * 80)
    print("CONFIG.TOML SNIPPET")
    print("=" * 80)
    print("\n[adaptive_noise]")
    print("enabled = true")
    print("default_multiplier = 1.0")
    print("\n[adaptive_noise.multipliers]")
    for source, mult in sorted(best_multipliers.items(), key=lambda x: x[1]):
        print(f'"{source}" = {mult:.2f}')

if __name__ == "__main__":
    main()
