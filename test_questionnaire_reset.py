import sys
import os
sys.path.insert(0, '/Users/brendanmullins/Projects/aitest/strem_process_anchor')

from datetime import datetime
from src.database import ProcessorStateDB
from src.reset_manager import ResetManager
import toml

# Load config
config = toml.load('config.toml')

# Get database
db = ProcessorStateDB(':memory:')

# Create a simple state with last weight
state = {
    'kalman_params': {'exists': True},
    'last_raw_weight': 91.26,
    'last_accepted_weight': 91.26,
    'last_timestamp': datetime(2025, 1, 27, 8, 55, 19),
    'last_accepted_timestamp': datetime(2025, 1, 27, 8, 55, 19)
}

# Test questionnaire measurement
weight = 95.25432
timestamp = datetime(2025, 1, 28, 0, 0, 0)
source = 'internal-questionnaire'

# Check if reset triggers
reset_type = ResetManager.should_trigger_reset(
    state, weight, timestamp, source, config
)

print(f"Source: {source}")
print(f"Previous weight: {state['last_raw_weight']} kg")
print(f"New weight: {weight} kg")
print(f"Weight change: {abs(weight - state['last_raw_weight']):.2f} kg")
print(f"Soft reset min change: {config.get('kalman', {}).get('reset', {}).get('soft', {}).get('min_weight_change_kg', 5)} kg")
print(f"Reset triggered: {reset_type}")
print()

# Now test with retrospective-replay source
source = 'retrospective-replay'
reset_type2 = ResetManager.should_trigger_reset(
    state, weight, timestamp, source, config
)
print(f"With source='retrospective-replay': Reset = {reset_type2}")
