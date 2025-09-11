"""Test to reproduce the rejection message discrepancy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
import toml

def test_rejection_message_bug():
    """Reproduce the 8.2kg vs 51.5kg discrepancy."""
    
    # Load configs
    with open('config.toml', 'r') as f:
        configs = toml.load(f)
    
    processing_config = configs['processing']
    kalman_config = configs['kalman']
    
    # Clear database
    db = ProcessorStateDB()
    db.states = {}
    
    user_id = "test_user"
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    
    # Initialize with normal weights around 83kg
    weights = [82.5, 83.0, 82.8, 83.2, 82.9, 83.1, 82.7, 83.0, 82.6, 83.3]
    
    print("Initializing with normal weights...")
    for i, weight in enumerate(weights):
        timestamp = base_time + timedelta(hours=i*24)
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="scale",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            print(f"  {i+1}. {weight}kg → Accepted (filtered: {result['filtered_weight']:.1f}kg)")
    
    # Get current state
    state = db.get_state(user_id)
    print(f"\nCurrent state:")
    print(f"  last_raw_weight: {state.get('last_raw_weight', 'Not set')}")
    if state.get('last_state') is not None:
        import numpy as np
        last_state = state['last_state']
        if isinstance(last_state, np.ndarray):
            if len(last_state.shape) > 1:
                print(f"  last_state (filtered): {last_state[-1][0]:.1f}kg")
            else:
                print(f"  last_state (filtered): {last_state[0]:.1f}kg")
    
    # Now submit an erroneous 31.5kg weight
    print("\nSubmitting erroneous weight...")
    timestamp = base_time + timedelta(hours=len(weights)*24 + 0.6)
    result = WeightProcessor.process_weight(
        user_id=user_id,
        weight=31.5,
        timestamp=timestamp,
        source="scale",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    print(f"Result for 31.5kg:")
    print(f"  Accepted: {result['accepted']}")
    if not result['accepted']:
        print(f"  Rejection reason: {result['reason']}")
        
        # Calculate what the change should be
        if 'last_raw_weight' in state:
            actual_change = abs(31.5 - state['last_raw_weight'])
            print(f"  Expected change: {actual_change:.1f}kg (31.5 - {state['last_raw_weight']:.1f})")
        
        # Parse the rejection message to see what it says
        import re
        match = re.search(r'Change of ([\d.]+)kg', result['reason'])
        if match:
            reported_change = float(match.group(1))
            print(f"  Reported change: {reported_change}kg")

if __name__ == "__main__":
    test_rejection_message_bug()