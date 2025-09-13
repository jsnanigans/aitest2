"""
Simple evolutionary algorithm to optimize adaptive noise multipliers.
Uses Kalman filter simulation without full processor overhead.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import json
import random
from pykalman import KalmanFilter

class SimpleMultiplierOptimizer:
    """Optimize adaptive noise multipliers using simplified Kalman simulation."""
    
    def __init__(self, data_file: str):
        """Initialize optimizer with data."""
        self.data_file = data_file
        self.load_data()
        
    def load_data(self):
        """Load and prepare data for optimization."""
        print(f"Loading data from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        
        # Rename columns
        df['timestamp'] = pd.to_datetime(df['effectiveDateTime'])
        df['source'] = df['source_type']
        
        # Get source statistics
        self.sources = df['source'].value_counts().to_dict()
        print(f"\nSource distribution:")
        for source, count in sorted(self.sources.items(), key=lambda x: -x[1])[:10]:
            print(f"  {source}: {count:,} measurements")
        
        # Sample diverse users
        print("\nSampling test users...")
        user_stats = df.groupby('user_id').agg({
            'source': 'nunique',
            'weight': 'count'
        }).rename(columns={'source': 'num_sources', 'weight': 'num_measurements'})
        
        # Get users with multiple sources and enough measurements
        good_users = user_stats[
            (user_stats['num_sources'] >= 2) & 
            (user_stats['num_measurements'] >= 10)
        ].index.tolist()
        
        # Sample up to 100 users
        sample_users = random.sample(good_users, min(100, len(good_users)))
        
        # Get their data
        self.test_data = df[df['user_id'].isin(sample_users)].copy()
        self.test_data = self.test_data.sort_values(['user_id', 'timestamp'])
        
        print(f"Selected {len(sample_users)} users with {len(self.test_data)} measurements")
        
        # Group by user
        self.user_sequences = {}
        for user_id, user_df in self.test_data.groupby('user_id'):
            self.user_sequences[user_id] = {
                'weights': user_df['weight'].values,
                'sources': user_df['source'].values,
                'timestamps': user_df['timestamp'].values
            }
    
    def simulate_kalman(self, weights: np.ndarray, sources: List[str], 
                       multipliers: Dict[str, float], base_noise: float = 3.49) -> Dict:
        """Simulate Kalman filtering with adaptive noise."""
        if len(weights) < 2:
            return {'error': float('inf')}
        
        # Initialize arrays
        filtered_weights = []
        confidences = []
        innovations = []
        
        # Initial state
        state_mean = np.array([weights[0], 0])  # [weight, trend]
        state_cov = np.array([[1.0, 0], [0, 0.001]])
        
        # Process each measurement
        for i, (weight, source) in enumerate(zip(weights, sources)):
            # Get adaptive noise for this source
            multiplier = multipliers.get(source, 1.0)
            obs_noise = base_noise * multiplier
            
            # Create Kalman filter with adapted noise
            kf = KalmanFilter(
                transition_matrices=np.array([[1, 1], [0, 1]]),  # Simple trend model
                observation_matrices=np.array([[1, 0]]),
                transition_covariance=np.array([[0.016, 0], [0, 0.0001]]),
                observation_covariance=np.array([[obs_noise]]),
                initial_state_mean=state_mean,
                initial_state_covariance=state_cov
            )
            
            # Update
            observation = np.array([weight])
            new_mean, new_cov = kf.filter_update(
                state_mean, state_cov, observation
            )
            
            # Calculate metrics
            predicted_weight = state_mean[0]
            innovation = weight - predicted_weight
            innovation_var = state_cov[0, 0] + obs_noise
            normalized_innovation = abs(innovation) / np.sqrt(innovation_var) if innovation_var > 0 else 0
            
            # Simple confidence calculation
            confidence = np.exp(-0.5 * normalized_innovation ** 2)
            
            # Store results
            filtered_weights.append(new_mean[0])
            confidences.append(confidence)
            innovations.append(abs(innovation))
            
            # Update state
            state_mean = new_mean
            state_cov = new_cov
        
        # Calculate overall metrics
        filtered_weights = np.array(filtered_weights)
        
        # Prediction error (how well we predict next measurement)
        if len(weights) > 1:
            pred_errors = np.abs(filtered_weights[:-1] - weights[1:])
            avg_error = np.mean(pred_errors)
        else:
            avg_error = 0
        
        # Stability (smoothness)
        if len(filtered_weights) > 1:
            stability = np.std(np.diff(filtered_weights))
        else:
            stability = 0
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        return {
            'error': avg_error,
            'stability': stability,
            'confidence': avg_confidence,
            'innovations': np.mean(innovations)
        }
    
    def evaluate_multipliers(self, multipliers: Dict[str, float]) -> float:
        """Evaluate multipliers across all test users."""
        total_error = 0
        total_stability = 0
        total_confidence = 0
        user_count = 0
        
        for user_id, data in self.user_sequences.items():
            result = self.simulate_kalman(
                data['weights'], 
                data['sources'], 
                multipliers
            )
            
            if result['error'] != float('inf'):
                total_error += result['error']
                total_stability += result['stability']
                total_confidence += result['confidence']
                user_count += 1
        
        if user_count == 0:
            return float('inf')
        
        # Fitness function (lower is better)
        fitness = (
            (total_error / user_count) * 1.0 +  # Prediction accuracy
            (total_stability / user_count) * 0.3 +  # Smoothness
            (1.0 - total_confidence / user_count) * 0.5  # Confidence (inverted)
        )
        
        return fitness
    
    def optimize(self, generations: int = 30, population_size: int = 20):
        """Run evolutionary optimization."""
        print("\n" + "=" * 80)
        print("EVOLUTIONARY OPTIMIZATION OF ADAPTIVE MULTIPLIERS")
        print("=" * 80)
        
        # Get unique sources
        all_sources = list(self.sources.keys())
        
        # Initialize population
        print(f"\nInitializing population of {population_size} individuals...")
        population = []
        
        # Add current hardcoded values
        current = {
            'care-team-upload': 0.5,
            'patient-upload': 0.7,
            'internal-questionnaire': 0.8,
            'initial-questionnaire': 0.8,
            'patient-device': 1.0,
            'https://connectivehealth.io': 1.5,
            'https://api.iglucose.com': 3.0
        }
        
        # Fill missing sources
        for source in all_sources:
            if source not in current:
                current[source] = 1.0
        population.append(current)
        
        # Add some variations of current
        for i in range(3):
            variant = {}
            for source in all_sources:
                base = current.get(source, 1.0)
                variant[source] = max(0.3, min(5.0, base * random.uniform(0.7, 1.3)))
            population.append(variant)
        
        # Add random individuals
        while len(population) < population_size:
            individual = {}
            for source in all_sources:
                individual[source] = random.uniform(0.3, 5.0)
            population.append(individual)
        
        # Evolution
        best_fitness = float('inf')
        best_individual = None
        history = []
        
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate
            fitness_scores = []
            for i, ind in enumerate(population):
                fitness = self.evaluate_multipliers(ind)
                fitness_scores.append((fitness, ind))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = ind.copy()
                
                if i % 5 == 0:
                    print(f"  Evaluated {i+1}/{population_size} (best: {best_fitness:.4f})")
            
            history.append(best_fitness)
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[0])
            
            # Print generation stats
            gen_best = fitness_scores[0][0]
            gen_worst = fitness_scores[-1][0]
            gen_avg = np.mean([f for f, _ in fitness_scores])
            print(f"  Generation stats: best={gen_best:.4f}, avg={gen_avg:.4f}, worst={gen_worst:.4f}")
            
            # Selection and reproduction
            elite_size = max(2, population_size // 5)
            new_population = [ind for _, ind in fitness_scores[:elite_size]]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = random.choice([ind for _, ind in fitness_scores[:population_size//2]])
                parent2 = random.choice([ind for _, ind in fitness_scores[:population_size//2]])
                
                # Crossover
                child = {}
                for source in all_sources:
                    if random.random() < 0.5:
                        child[source] = parent1.get(source, 1.0)
                    else:
                        child[source] = parent2.get(source, 1.0)
                
                # Mutation
                for source in all_sources:
                    if random.random() < 0.2:  # 20% mutation rate
                        change = np.random.normal(0, 0.3)
                        child[source] = max(0.3, min(5.0, child[source] + change))
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual, best_fitness, history

def main():
    # Run optimization
    optimizer = SimpleMultiplierOptimizer("./data/2025-09-05_optimized.csv")
    
    best_multipliers, best_fitness, history = optimizer.optimize(
        generations=20,
        population_size=30
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\nBest fitness: {best_fitness:.4f}")
    
    print("\nOptimized multipliers (sorted by value):")
    for source, mult in sorted(best_multipliers.items(), key=lambda x: x[1]):
        count = optimizer.sources.get(source, 0)
        print(f"  {mult:.2f} - {source} ({count:,} measurements)")
    
    # Compare with current
    current = {
        'care-team-upload': 0.5,
        'patient-upload': 0.7,
        'internal-questionnaire': 0.8,
        'patient-device': 1.0,
        'https://connectivehealth.io': 1.5,
        'https://api.iglucose.com': 3.0
    }
    
    print("\nChanges from current values:")
    for source in sorted(best_multipliers.keys()):
        old = current.get(source, 1.0)
        new = best_multipliers[source]
        if abs(old - new) > 0.1:
            print(f"  {source}: {old:.2f} → {new:.2f} ({(new-old)/old*100:+.1f}%)")
    
    # Show fitness improvement
    print(f"\nFitness improvement over generations:")
    for i in range(0, len(history), 5):
        print(f"  Gen {i+1}: {history[i]:.4f}")
    
    # Save results
    results = {
        'optimized_multipliers': best_multipliers,
        'best_fitness': best_fitness,
        'fitness_history': history,
        'source_counts': optimizer.sources,
        'optimization_date': datetime.now().isoformat()
    }
    
    with open('test_output/optimal_multipliers.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_output/optimal_multipliers.json")
    
    # Generate config
    print("\n" + "=" * 80)
    print("SUGGESTED CONFIG.TOML ADDITION")
    print("=" * 80)
    print("\n[adaptive_noise]")
    print("# Enable adaptive measurement noise based on source reliability")
    print("enabled = true")
    print("")
    print("# Default multiplier for unknown sources")
    print("default_multiplier = 1.0")
    print("")
    print("[adaptive_noise.multipliers]")
    print("# Optimized via evolutionary algorithm")
    print(f"# Based on {len(optimizer.test_data)} measurements from {len(optimizer.user_sequences)} users")
    
    for source, mult in sorted(best_multipliers.items(), key=lambda x: x[1]):
        count = optimizer.sources.get(source, 0)
        if count > 100:  # Only include sources with enough data
            print(f'"{source}" = {mult:.2f}  # {count:,} measurements')

if __name__ == "__main__":
    main()
