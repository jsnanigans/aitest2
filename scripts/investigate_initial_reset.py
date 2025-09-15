#!/usr/bin/env python3

import sys
import pandas as pd
from datetime import datetime
import toml
from src.processor import WeightProcessor
from src.database import ProcessorDatabase
from src.reset_manager import ResetManager, ResetType

def investigate_initial_reset(user_id="091baa98-cf05-4399-b490-e24324f7607f"):
    """Investigate why initial reset isn't working for a specific user."""
    
    print(f"\n=== Investigating Initial Reset for User {user_id} ===\n")
    
    # Load config
    config = toml.load("config.toml")
    
    # Check initial reset config
    initial_config = config.get('kalman', {}).get('reset', {}).get('initial', {})
    print("Initial Reset Configuration:")
    for key, value in initial_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    csv_file = config['data']['csv_file']
    df = pd.read_csv(csv_file)
    
    # Filter for user
    user_data = df[df['user_id'] == user_id].copy()
    if user_data.empty:
        print(f"No data found for user {user_id}")
        return
    
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
    user_data = user_data.sort_values('timestamp')
    
    print(f"Found {len(user_data)} measurements for user")
    print(f"Date range: {user_data['timestamp'].min()} to {user_data['timestamp'].max()}")
    print()
    
    # Process first few measurements
    db = ProcessorDatabase()
    
    print("Processing first 5 measurements:")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(user_data.head(5).iterrows()):
        timestamp = row['timestamp']
        weight = row['weight']
        source = row.get('source', 'unknown')
        
        print(f"\nMeasurement {idx + 1}:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Weight: {weight:.1f} kg")
        print(f"  Source: {source}")
        
        # Get current state
        state = db.get_state(user_id)
        
        # Check if reset should trigger
        reset_type = ResetManager.should_trigger_reset(
            state, weight, timestamp, source, config
        )
        
        if reset_type:
            print(f"  Reset Type: {reset_type.value}")
            
            # Get reset parameters
            reset_params = ResetManager.get_reset_parameters(reset_type, config)
            print(f"  Reset Parameters:")
            for key, value in reset_params.items():
                print(f"    {key}: {value}")
        else:
            print(f"  Reset Type: None")
        
        # Process the measurement
        result = WeightProcessor.process_measurement(
            user_id, weight, timestamp, source, config
        )
        
        print(f"  Result: {'ACCEPTED' if result['accepted'] else 'REJECTED'}")
        if not result['accepted']:
            print(f"  Rejection Reason: {result.get('rejection_reason', 'unknown')}")
        
        # Check state after processing
        state = db.get_state(user_id)
        if state.get('reset_type'):
            print(f"  State Reset Type: {state['reset_type']}")
        if state.get('reset_parameters'):
            print(f"  Active Reset Parameters: {state['reset_parameters']}")
        
        # Check measurements since reset
        measurements_since = state.get('measurements_since_reset', 0)
        print(f"  Measurements Since Reset: {measurements_since}")
    
    print("\n" + "=" * 80)
    print("\nSummary:")
    
    # Check final state
    final_state = db.get_state(user_id)
    if final_state.get('reset_events'):
        print(f"Total Reset Events: {len(final_state['reset_events'])}")
        for event in final_state['reset_events']:
            print(f"  - {event.get('type', 'unknown')} at {event.get('timestamp')}")
    else:
        print("No reset events recorded")

if __name__ == "__main__":
    user_id = sys.argv[1] if len(sys.argv) > 1 else "091baa98-cf05-4399-b490-e24324f7607f"
    investigate_initial_reset(user_id)
