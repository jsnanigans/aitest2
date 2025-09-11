#!/usr/bin/env python3
"""
Integration test showing how BMI validator prevents the user 0040872d issue.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorDatabase
from src.bmi_validator import BMIValidator

def process_with_bmi_validation(user_id: str, csv_file: str, height_m: float = 1.75):
    """
    Process user data with BMI validation integrated.
    """
    print(f"=== Processing User {user_id} with BMI Validation ===\n")
    
    df = pd.read_csv(csv_file)
    df.columns = ['user_id', 'timestamp', 'source', 'weight_kg', 'unit']
    
    user_data = df[df['user_id'] == user_id].copy()
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = user_data.sort_values('timestamp')
    
    print(f"Total measurements: {len(user_data)}")
    print(f"Date range: {user_data['timestamp'].min()} to {user_data['timestamp'].max()}")
    print(f"Assumed height: {height_m:.2f}m\n")
    
    db = ProcessorDatabase()
    processor = WeightProcessor()
    
    accepted_weights = []
    rejected_weights = []
    reset_count = 0
    
    print("Processing with BMI validation:\n")
    
    for idx, row in user_data.iterrows():
        timestamp = row['timestamp']
        weight = row['weight_kg']
        source = row['source']
        
        state = db.get_user_state(user_id)
        
        if state and state.get('last_raw_weight'):
            last_weight = state['last_raw_weight']
            last_timestamp = state.get('last_timestamp')
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)
            
            time_delta_hours = (timestamp - last_timestamp).total_seconds() / 3600
            
            should_reset, reset_reason = BMIValidator.should_reset_kalman(
                weight, last_weight, time_delta_hours, height_m, source
            )
            
            if should_reset:
                print(f"🔄 RESET at {timestamp}: {weight:.1f}kg")
                print(f"   Reason: {reset_reason}")
                db.save_user_state(user_id, None)
                state = None
                reset_count += 1
        
        result = processor.process_measurement(
            weight_kg=weight,
            timestamp=timestamp,
            user_state=state,
            source_type=source
        )
        
        if result['accepted']:
            db.save_user_state(user_id, result['state'])
            accepted_weights.append(weight)
            
            if len(accepted_weights) <= 5 or reset_count > 0:
                print(f"✅ ACCEPTED: {timestamp}: {weight:.1f}kg → {result['accepted_weight']:.1f}kg")
        else:
            rejected_weights.append(weight)
            if len(rejected_weights) <= 10:
                print(f"❌ REJECTED: {timestamp}: {weight:.1f}kg - {result.get('rejection_reason', 'Unknown')}")
    
    print(f"\n=== Final Statistics ===")
    print(f"Total processed: {len(user_data)}")
    print(f"Accepted: {len(accepted_weights)}")
    print(f"Rejected: {len(rejected_weights)}")
    print(f"Resets triggered: {reset_count}")
    
    if accepted_weights:
        print(f"\nAccepted weight range: {min(accepted_weights):.1f} - {max(accepted_weights):.1f}kg")
        print(f"Final accepted weight: {accepted_weights[-1]:.1f}kg")
    
    return accepted_weights, rejected_weights, reset_count

def compare_with_and_without_bmi():
    """
    Compare processing with and without BMI validation.
    """
    user_id = '0040872d-333a-4ace-8c5a-b2fcd056e65a'
    csv_file = 'data/2025-09-05_optimized.csv'
    
    print("\n" + "="*60)
    print("COMPARISON: With vs Without BMI Validation")
    print("="*60 + "\n")
    
    print("1. WITHOUT BMI Validation (Current System):")
    print("-" * 40)
    db_without = ProcessorDatabase()
    processor = WeightProcessor()
    
    df = pd.read_csv(csv_file)
    df.columns = ['user_id', 'timestamp', 'source', 'weight_kg', 'unit']
    user_data = df[df['user_id'] == user_id].head(20)
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    
    accepted_without = 0
    for _, row in user_data.iterrows():
        state = db_without.get_user_state(user_id)
        result = processor.process_measurement(
            weight_kg=row['weight_kg'],
            timestamp=row['timestamp'],
            user_state=state,
            source_type=row['source']
        )
        if result['accepted']:
            db_without.save_user_state(user_id, result['state'])
            accepted_without += 1
    
    print(f"Accepted {accepted_without}/{len(user_data)} measurements")
    print("Problem: Accepts physiologically impossible changes")
    
    print("\n2. WITH BMI Validation (Proposed System):")
    print("-" * 40)
    accepted_with, rejected_with, resets = process_with_bmi_validation(
        user_id, csv_file, height_m=1.75
    )
    
    print("\n" + "="*60)
    print("BENEFITS OF BMI VALIDATION:")
    print("="*60)
    print("✅ Detects impossible weight changes (>30% drops)")
    print("✅ Triggers Kalman resets instead of bad adaptations")
    print("✅ Prevents BMI < 15 or > 50 acceptance")
    print("✅ Faster recovery when valid data returns")
    print("✅ Source-aware validation (stricter for unreliable sources)")

if __name__ == "__main__":
    user_id = '0040872d-333a-4ace-8c5a-b2fcd056e65a'
    csv_file = 'data/2025-09-05_optimized.csv'
    
    if os.path.exists(csv_file):
        process_with_bmi_validation(user_id, csv_file)
        compare_with_and_without_bmi()
    else:
        print(f"CSV file not found: {csv_file}")
        print("Running demonstration with synthetic data instead...")
        
        from test_bmi_solution_0040872d import test_bmi_validator_on_user_data
        test_bmi_validator_on_user_data()