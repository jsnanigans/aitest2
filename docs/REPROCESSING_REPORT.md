# Weight Stream Reprocessing System - Analysis Report

## Executive Summary

Analysis of weight stream data reveals critical issues requiring a dedicated reprocessing system:
1. **Extreme outliers** in daily batches (ranges up to 6,103 kg!)
2. **Retroactive data additions** with questionnaire data arriving after device data
3. **Multiple readings per day** damaging Kalman filter accuracy (8% of users affected)

## Data Analysis Findings

### Issue 1: Extreme Multiple Daily Values

Found cases with massive discrepancies in same-day readings:

**Most Extreme Case:**
- User: EB61194E... on 2025-03-06
- Range: 6,103 kg (weights: 6226.0 kg vs 123.0 kg)
- Sources: patient-device vs api.iglucose.com
- **Impact**: First value (6226 kg) would completely destroy Kalman filter state

**High-Volume Case:**
- User: EB61194E... on 2025-03-25
- 39 readings in one day
- Range: 97 kg (7 kg to 104 kg)
- All from patient-device
- **Impact**: First-come-first-served would pick arbitrary values

### Issue 2: Source Ordering Violations

Found multiple cases where device data precedes questionnaire data:

**Example:**
- User: 1A4E17CA...
- Questionnaire date: 2025-01-29
- Earlier device readings: 70 entries starting 2025-01-07
- **Impact**: Initial baseline established from device data, questionnaire ignored

### Issue 3: Kalman Filter Damage Assessment

Analysis shows the current first-come-first-served approach causes significant errors:
- Days with >3 kg range between values: Common
- Median weight often differs from first weight by >2 kg
- Outliers can shift Kalman state by 10+ kg in extreme cases

## Proposed Architecture: Dual-Processor System

### 1. Primary Processor (Real-time)
- Current `processor.py` continues handling real-time streaming
- Provides immediate feedback to users
- Maintains forward-only processing

### 2. Reprocessor (Batch/Delayed)
- New `reprocessor.py` for retroactive corrections
- Runs on schedule or triggered by events
- Can rewind and replay from any point

## Implementation Design

### Core Components

```python
# reprocessor.py - New stateless reprocessor

class WeightReprocessor:
    """Handles retroactive reprocessing of weight data."""
    
    @staticmethod
    def reprocess_from_date(
        user_id: str,
        start_date: datetime,
        measurements: List[Dict],
        processing_config: Dict,
        kalman_config: Dict
    ) -> Dict:
        """Reprocess all measurements from a specific date."""
        pass
    
    @staticmethod
    def process_daily_batch(
        user_id: str,
        date: date,
        measurements: List[Dict],
        processing_config: Dict,
        kalman_config: Dict
    ) -> Dict:
        """Process a day's measurements with outlier detection."""
        pass
    
    @staticmethod
    def select_best_measurements(
        measurements: List[Dict],
        max_deviation: float = 2.0
    ) -> List[Dict]:
        """Select the most consistent measurements from a batch."""
        pass
```

### State Management Extensions

```python
# processor_database.py - Extended with snapshot capability

class ProcessorDatabase:
    """Extended database with snapshot support."""
    
    def create_snapshot(self, user_id: str, timestamp: datetime) -> str:
        """Create a snapshot of current state."""
        pass
    
    def restore_snapshot(self, user_id: str, snapshot_id: str) -> Dict:
        """Restore state from snapshot."""
        pass
    
    def get_state_at_date(self, user_id: str, date: datetime) -> Dict:
        """Get the state as it was at a specific date."""
        pass
```

### Daily Batch Processing Algorithm

```python
def process_daily_batch(measurements):
    """Smart selection from multiple daily measurements."""
    
    # Step 1: Group by source reliability
    sources_priority = {
        'internal-questionnaire': 1,
        'https://api.iglucose.com': 2,
        'patient-device': 3,
        'patient-upload': 4
    }
    
    # Step 2: Statistical outlier detection
    if len(measurements) > 2:
        median = np.median([m['weight'] for m in measurements])
        mad = np.median(np.abs([m['weight'] - median for m in measurements]))
        threshold = median + (3 * mad)
        
        # Filter extreme outliers
        valid = [m for m in measurements 
                if abs(m['weight'] - median) < threshold]
    else:
        valid = measurements
    
    # Step 3: Source-based selection
    if len(valid) > 1:
        # Prefer higher priority sources
        valid.sort(key=lambda m: sources_priority.get(m['source'], 999))
        
    # Step 4: Consistency check with recent history
    # (Compare with last known good weight)
    
    return valid[0] if valid else None
```

## Trigger Conditions for Reprocessing

### 1. Retroactive Data Addition
```python
def check_retroactive_addition(user_id, new_measurement):
    """Detect if measurement is retroactive."""
    last_processed = db.get_last_timestamp(user_id)
    if new_measurement['timestamp'] < last_processed:
        # Trigger reprocessing from new_measurement date
        return True
    return False
```

### 2. Daily Batch Processing (Scheduled)
```python
def daily_reprocessing_job():
    """Run at end of day to clean up data."""
    today = datetime.now().date()
    
    for user_id in get_active_users():
        measurements = get_measurements_for_date(user_id, today)
        
        if len(measurements) > 1:
            # Multiple measurements found
            best = select_best_measurements(measurements)
            
            if best != measurements[0]:  # First wasn't best
                # Reprocess from start of day
                reprocess_from_date(user_id, today)
```

### 3. Manual Trigger
```python
def handle_data_correction_request(user_id, date, reason):
    """Handle manual reprocessing requests."""
    # Create snapshot for rollback
    snapshot_id = db.create_snapshot(user_id, datetime.now())
    
    try:
        # Reprocess from requested date
        result = reprocess_from_date(user_id, date)
        
        if not validate_reprocessing(result):
            # Rollback if validation fails
            db.restore_snapshot(user_id, snapshot_id)
            
    except Exception as e:
        db.restore_snapshot(user_id, snapshot_id)
        raise
```

## Migration Strategy

### Phase 1: Parallel Processing (Week 1-2)
1. Implement reprocessor without affecting primary flow
2. Run in shadow mode, log differences
3. Validate improvements on historical data

### Phase 2: Selective Activation (Week 3-4)
1. Enable for users with known issues
2. Monitor impact on accuracy
3. Tune outlier detection thresholds

### Phase 3: Full Deployment (Week 5-6)
1. Enable daily batch processing for all users
2. Implement retroactive detection
3. Add monitoring and alerting

## Testing Strategy

### Test Cases Required

1. **Extreme Outlier Handling**
   - Input: [6226.0, 123.0] kg on same day
   - Expected: Select 123.0 kg, flag 6226.0 as error

2. **Multiple Valid Readings**
   - Input: [85.2, 85.5, 85.3] kg
   - Expected: Use median (85.3 kg)

3. **Source Priority**
   - Input: questionnaire=80kg, device=75kg
   - Expected: Prefer questionnaire (80kg)

4. **Retroactive Processing**
   - Add data from 10 days ago
   - Verify Kalman replays correctly from that point

5. **State Consistency**
   - Process, snapshot, reprocess
   - Verify state management integrity

## Performance Considerations

- **State Snapshots**: ~1KB per snapshot, store last 30 days
- **Reprocessing Cost**: O(n) where n = measurements since date
- **Daily Batch**: Process overnight during low activity
- **Memory**: Keep minimal (stateless design maintained)

## Risk Mitigation

1. **Always snapshot before reprocessing**
2. **Validate results match expected patterns**
3. **Log all reprocessing events for audit**
4. **Implement gradual rollout**
5. **Monitor Kalman stability metrics**

## Success Metrics

- Reduction in outlier acceptance: >90%
- Kalman prediction error: <1kg RMSE
- Retroactive correction success rate: >95%
- Processing time per daily batch: <100ms
- User weight tracking accuracy improvement: >20%

## Next Steps

1. **Implement `reprocessor.py`** with core algorithms
2. **Extend `processor_database.py`** with snapshot support
3. **Create comprehensive test suite** for edge cases
4. **Build monitoring dashboard** for reprocessing events
5. **Document API for manual corrections**

## Conclusion

The data analysis clearly shows the need for a reprocessing system. The proposed dual-processor architecture maintains the simplicity of the current system while adding crucial data quality capabilities. The stateless design ensures scalability and the snapshot system provides safety for retroactive changes.