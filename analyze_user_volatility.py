#!/usr/bin/env python3
"""
Analyze why user 1e87d3ab-20b1-479d-ad4d-8986e1af38da has volatile Kalman output
"""

import sys
sys.path.insert(0, '.')

import csv
from datetime import datetime
import numpy as np
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
import tomllib
import matplotlib.pyplot as plt

def main():
    user_id = "1e87d3ab-20b1-479d-ad4d-8986e1af38da"
    csv_path = "./data/2025-09-05_optimized.csv"
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    measurements = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == user_id:
                weight_str = row.get("weight", "").strip()
                if weight_str and weight_str.upper() != "NULL":
                    try:
                        weight = float(weight_str)
                        timestamp = datetime.strptime(row.get('effectiveDateTime'), "%Y-%m-%d %H:%M:%S")
                        measurements.append((timestamp, weight))
                    except:
                        continue
    
    measurements.sort(key=lambda x: x[0])
    print(f"Found {len(measurements)} valid measurements for user {user_id}")
    
    # Analyze raw data stability
    weights = [w for _, w in measurements]
    print(f"\nRaw data statistics:")
    print(f"  Mean: {np.mean(weights):.2f} kg")
    print(f"  Std Dev: {np.std(weights):.2f} kg")
    print(f"  Min: {np.min(weights):.2f} kg")
    print(f"  Max: {np.max(weights):.2f} kg")
    print(f"  Range: {np.max(weights) - np.min(weights):.2f} kg")
    
    # Check for time gaps
    print(f"\nAnalyzing time gaps:")
    large_gaps = []
    for i in range(1, min(len(measurements), 20)):
        gap_days = (measurements[i][0] - measurements[i-1][0]).days
        if gap_days > 7:
            large_gaps.append((i, gap_days, measurements[i-1][0], measurements[i][0]))
            print(f"  Gap at position {i}: {gap_days} days ({measurements[i-1][0].date()} to {measurements[i][0].date()})")
    
    # Process through Kalman and track state evolution
    db = ProcessorStateDB()
    results = []
    states = []
    covariances = []
    
    print(f"\nProcessing through Kalman filter:")
    print(f"Kalman config:")
    print(f"  observation_covariance: {config['kalman']['observation_covariance']}")
    print(f"  transition_covariance_weight: {config['kalman']['transition_covariance_weight']}")
    print(f"  transition_covariance_trend: {config['kalman']['transition_covariance_trend']}")
    print(f"  initial_variance: {config['kalman']['initial_variance']}")
    print(f"  reset_gap_days: {config['kalman']['reset_gap_days']}")
    
    for i, (timestamp, weight) in enumerate(measurements[:50]):  # Process first 50
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="csv",
            processing_config=config["processing"],
            kalman_config=config["kalman"],
            db=db
        )
        
        if result:
            results.append(result)
            
            # Get state after processing
            state = db.get_state(user_id)
            if state and state.get('last_state') is not None:
                last_state = state['last_state']
                if len(last_state.shape) > 1:
                    current_state = last_state[-1]
                else:
                    current_state = last_state
                states.append(current_state)
                
                # Track covariance
                if state.get('last_covariance') is not None:
                    last_cov = state['last_covariance']
                    if len(last_cov.shape) > 2:
                        current_cov = last_cov[-1]
                    else:
                        current_cov = last_cov
                    covariances.append(current_cov[0, 0])  # Weight variance
            
            # Check for anomalies
            if i > 0 and 'filtered_weight' in result:
                raw = result['raw_weight']
                filtered = result['filtered_weight']
                diff = abs(raw - filtered)
                if diff > 2.0:  # Large deviation
                    print(f"\n  Measurement {i+1} ({timestamp.date()}):")
                    print(f"    Raw: {raw:.2f} kg")
                    print(f"    Filtered: {filtered:.2f} kg")
                    print(f"    Difference: {diff:.2f} kg")
                    print(f"    Innovation: {result.get('innovation', 0):.2f}")
                    print(f"    Normalized Innovation: {result.get('normalized_innovation', 0):.2f}")
                    print(f"    Confidence: {result.get('confidence', 0):.2%}")
                    if i < len(covariances):
                        print(f"    Weight variance: {covariances[i]:.4f}")
                    
                    # Check previous measurement
                    if i > 0:
                        prev_result = results[i-1]
                        gap_days = (timestamp - prev_result['timestamp']).days
                        if gap_days > 30:
                            print(f"    *** LARGE GAP: {gap_days} days since previous ***")
    
    # Plot analysis
    if len(results) > 10:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Raw vs Filtered
        timestamps = [r['timestamp'] for r in results]
        raw_weights = [r['raw_weight'] for r in results]
        filtered_weights = [r['filtered_weight'] for r in results if 'filtered_weight' in r]
        
        axes[0].plot(timestamps[:len(filtered_weights)], filtered_weights, 'b-', label='Kalman Filtered', linewidth=2)
        axes[0].scatter(timestamps, raw_weights, c='green', alpha=0.6, label='Raw Data', s=30)
        axes[0].set_ylabel('Weight (kg)')
        axes[0].set_title(f'User {user_id[:8]}... - Raw vs Filtered')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Innovation (difference between raw and predicted)
        innovations = [r['innovation'] for r in results if 'innovation' in r]
        axes[1].plot(timestamps[:len(innovations)], innovations, 'r-', linewidth=1)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Innovation (kg)')
        axes[1].set_title('Innovation (Raw - Predicted)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Covariance evolution
        if covariances:
            axes[2].plot(timestamps[:len(covariances)], covariances, 'purple', linewidth=1)
            axes[2].set_ylabel('Weight Variance')
            axes[2].set_title('Kalman Filter Weight Variance Evolution')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'output/user_{user_id[:8]}_analysis.png', dpi=150)
        print(f"\nAnalysis plot saved to output/user_{user_id[:8]}_analysis.png")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"User has relatively stable raw data (std dev: {np.std(weights):.2f} kg)")
    print(f"Number of large gaps (>7 days): {len(large_gaps)}")
    if large_gaps:
        print(f"Largest gap: {max(g[1] for g in large_gaps)} days")
    
    # Check for pattern in volatility
    if len(results) > 20:
        recent_innovations = [abs(r['innovation']) for r in results[-20:] if 'innovation' in r]
        if recent_innovations:
            print(f"Recent average absolute innovation: {np.mean(recent_innovations):.2f} kg")

if __name__ == "__main__":
    main()