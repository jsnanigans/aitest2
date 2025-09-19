# Plan: Provisional Acceptance Mechanism

## Problem Statement

Current system makes binary accept/reject decisions immediately. Real data shows cases where:
- Measurements after long gaps could be valid or errors (can't determine immediately)
- Source produces suspicious but possibly correct values
- Need subsequent measurements to validate uncertain values

Example: After 137-day gap, user reports 139.6kg (was 33.8kg). System must accept or reject immediately without context to validate.

## Objectives

1. Allow temporary acceptance of uncertain measurements
2. Mark measurements for later validation
3. Automatically confirm or revoke based on subsequent data
4. Maintain state consistency during provisional period
5. Provide rollback capability if provisional acceptance was wrong

## Implementation Design

### 1. Provisional State Manager

```python
# src/processing/provisional_manager.py

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class ProvisionalStatus(Enum):
    """Status of provisional measurements."""
    PENDING = "pending"          # Awaiting confirmation
    CONFIRMED = "confirmed"       # Validated by subsequent measurements
    REJECTED = "rejected"         # Invalidated by subsequent measurements
    EXPIRED = "expired"          # Confirmation window passed without validation

class ProvisionalManager:
    """Manages provisional acceptance of uncertain measurements."""

    def __init__(self, db, config=None):
        self.db = db
        self.config = config or {}

        # Configuration
        self.confirmation_window = timedelta(
            hours=config.get('confirmation_window_hours', 72)
        )
        self.min_confirming_measurements = config.get('min_confirming_measurements', 3)
        self.confirmation_threshold = config.get('confirmation_threshold', 0.7)

        # In-memory tracking (also persisted to DB)
        self.provisional_measurements = {}  # user_id -> list of provisional

    def accept_provisional(self, user_id: str, measurement: Dict[str, Any],
                          uncertainty_reason: str) -> Dict[str, Any]:
        """
        Accept a measurement provisionally.

        Args:
            user_id: User identifier
            measurement: Measurement data
            uncertainty_reason: Why this is uncertain

        Returns:
            Provisional acceptance record
        """
        provisional_record = {
            'measurement_id': self._generate_id(),
            'user_id': user_id,
            'timestamp': measurement['timestamp'],
            'weight': measurement['weight'],
            'source': measurement['source'],
            'status': ProvisionalStatus.PENDING,
            'uncertainty_reason': uncertainty_reason,
            'confidence': measurement.get('confidence', 0.5),
            'accepted_at': datetime.now(),
            'expires_at': datetime.now() + self.confirmation_window,
            'confirming_measurements': [],
            'state_snapshot': self.db.get_state_snapshot(user_id)
        }

        # Track provisionally
        if user_id not in self.provisional_measurements:
            self.provisional_measurements[user_id] = []
        self.provisional_measurements[user_id].append(provisional_record)

        # Save to database
        self.db.save_provisional_measurement(provisional_record)

        # Update user state with provisional marker
        state = self.db.get_state(user_id)
        state['has_provisional'] = True
        state['provisional_weight'] = measurement['weight']
        self.db.save_state(user_id, state)

        return {
            'accepted': True,
            'provisional': True,
            'provisional_id': provisional_record['measurement_id'],
            'expires_at': provisional_record['expires_at'],
            'reason': uncertainty_reason,
            'confirmation_needed': self.min_confirming_measurements
        }

    def check_confirmations(self, user_id: str, new_measurement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if new measurement confirms or rejects provisional measurements.

        Args:
            user_id: User identifier
            new_measurement: New measurement to check against

        Returns:
            List of confirmation results
        """
        if user_id not in self.provisional_measurements:
            return []

        results = []
        pending = [p for p in self.provisional_measurements[user_id]
                  if p['status'] == ProvisionalStatus.PENDING]

        for provisional in pending:
            # Check if expired
            if datetime.now() > provisional['expires_at']:
                self._expire_provisional(provisional)
                results.append({
                    'provisional_id': provisional['measurement_id'],
                    'action': 'expired',
                    'reason': 'confirmation_window_passed'
                })
                continue

            # Calculate confirmation score
            confirmation_score = self._calculate_confirmation_score(
                provisional, new_measurement
            )

            # Add to confirming measurements
            provisional['confirming_measurements'].append({
                'timestamp': new_measurement['timestamp'],
                'weight': new_measurement['weight'],
                'score': confirmation_score
            })

            # Check if we have enough confirmations
            if len(provisional['confirming_measurements']) >= self.min_confirming_measurements:
                avg_score = sum(m['score'] for m in provisional['confirming_measurements']) / len(provisional['confirming_measurements'])

                if avg_score >= self.confirmation_threshold:
                    self._confirm_provisional(provisional)
                    results.append({
                        'provisional_id': provisional['measurement_id'],
                        'action': 'confirmed',
                        'confidence': avg_score
                    })
                else:
                    self._reject_provisional(provisional)
                    results.append({
                        'provisional_id': provisional['measurement_id'],
                        'action': 'rejected',
                        'confidence': avg_score,
                        'rollback_required': True
                    })

        return results

    def _calculate_confirmation_score(self, provisional: Dict[str, Any],
                                     new_measurement: Dict[str, Any]) -> float:
        """
        Calculate how well new measurement confirms provisional.

        Score based on:
        - Weight similarity
        - Trend consistency
        - Source reliability
        """
        score = 0.0
        weights = []

        # Weight similarity (40% weight)
        weight_diff = abs(new_measurement['weight'] - provisional['weight']) / provisional['weight']
        weight_score = max(0, 1.0 - weight_diff / 0.2)  # 20% difference = 0 score
        score += weight_score * 0.4
        weights.append(('weight_similarity', weight_score))

        # Trend consistency (30% weight)
        if provisional['state_snapshot']:
            expected_trend = self._calculate_expected_trend(provisional['state_snapshot'])
            actual_trend = (new_measurement['weight'] - provisional['weight']) / max(1, (new_measurement['timestamp'] - provisional['timestamp']).days)
            trend_diff = abs(actual_trend - expected_trend)
            trend_score = max(0, 1.0 - trend_diff / 2.0)
            score += trend_score * 0.3
            weights.append(('trend_consistency', trend_score))

        # Source reliability (30% weight)
        source_score = self._get_source_reliability_score(new_measurement['source'])
        score += source_score * 0.3
        weights.append(('source_reliability', source_score))

        return score

    def _confirm_provisional(self, provisional: Dict[str, Any]):
        """Confirm a provisional measurement as valid."""
        provisional['status'] = ProvisionalStatus.CONFIRMED
        provisional['confirmed_at'] = datetime.now()

        # Update database
        self.db.update_provisional_status(
            provisional['measurement_id'],
            ProvisionalStatus.CONFIRMED
        )

        # Remove provisional marker from state
        state = self.db.get_state(provisional['user_id'])
        state['has_provisional'] = False
        if 'provisional_weight' in state:
            del state['provisional_weight']
        self.db.save_state(provisional['user_id'], state)

    def _reject_provisional(self, provisional: Dict[str, Any]):
        """Reject a provisional measurement and trigger rollback."""
        provisional['status'] = ProvisionalStatus.REJECTED
        provisional['rejected_at'] = datetime.now()

        # Restore state from snapshot
        if provisional['state_snapshot']:
            self.db.restore_state(provisional['user_id'], provisional['state_snapshot'])

        # Update database
        self.db.update_provisional_status(
            provisional['measurement_id'],
            ProvisionalStatus.REJECTED
        )

        # Trigger reprocessing of measurements after provisional
        self._trigger_reprocessing(provisional)

    def _expire_provisional(self, provisional: Dict[str, Any]):
        """Expire a provisional measurement (treat as soft confirmation)."""
        provisional['status'] = ProvisionalStatus.EXPIRED
        provisional['expired_at'] = datetime.now()

        # For expired, we keep the measurement but lower confidence
        self.db.update_measurement_confidence(
            provisional['user_id'],
            provisional['timestamp'],
            confidence=0.4  # Lower confidence for unconfirmed
        )
```

### 2. Provisional Decision Logic

```python
class ProvisionalDecisionMaker:
    """Decides when to use provisional acceptance."""

    def __init__(self, config=None):
        self.config = config or {}

        # Thresholds for provisional acceptance
        self.quality_score_range = (0.3, 0.6)  # Between these scores
        self.gap_threshold_days = 30
        self.change_threshold = 0.15  # 15% change

    def should_accept_provisionally(self, measurement, context):
        """
        Determine if measurement should be provisionally accepted.

        Args:
            measurement: Current measurement
            context: User context including history

        Returns:
            tuple: (should_accept_provisionally, reason)
        """
        quality_score = measurement.get('quality_score', 0.5)

        # Case 1: After long gap
        if context.get('days_since_last') > self.gap_threshold_days:
            if self.quality_score_range[0] <= quality_score <= self.quality_score_range[1]:
                return True, 'long_gap_uncertainty'

        # Case 2: Large change but not impossible
        if context.get('last_weight'):
            change = abs(measurement['weight'] - context['last_weight']) / context['last_weight']
            if self.change_threshold < change < 0.3:  # Between 15% and 30%
                return True, 'large_change_uncertainty'

        # Case 3: Suspicious source but plausible value
        suspicious_sources = ['iglucose.com', 'manual-entry']
        if measurement['source'] in suspicious_sources:
            if self.quality_score_range[0] <= quality_score <= self.quality_score_range[1]:
                return True, 'suspicious_source'

        # Case 4: Conflicting with recent trend
        if context.get('trend_violation'):
            if quality_score > 0.4:
                return True, 'trend_violation'

        return False, None
```

### 3. Integration with Main Processing

```python
# Enhanced processor.py

def process_measurement(user_id, weight, timestamp, source, config, unit, db):
    """Process measurement with provisional acceptance capability."""

    # Regular validation...
    quality_score = calculate_quality_score(...)

    # Check for provisional decision
    decision_maker = ProvisionalDecisionMaker(config)
    context = {
        'last_weight': db.get_last_weight(user_id),
        'days_since_last': calculate_days_since_last(user_id, timestamp),
        'trend_violation': check_trend_violation(weight, db.get_trend(user_id))
    }

    should_provisional, reason = decision_maker.should_accept_provisionally(
        {'weight': weight, 'source': source, 'quality_score': quality_score},
        context
    )

    if should_provisional:
        # Accept provisionally
        provisional_mgr = ProvisionalManager(db, config)
        result = provisional_mgr.accept_provisional(
            user_id,
            {'weight': weight, 'timestamp': timestamp, 'source': source},
            reason
        )
        return result

    # Check if this confirms any provisional measurements
    provisional_mgr = ProvisionalManager(db, config)
    confirmations = provisional_mgr.check_confirmations(
        user_id,
        {'weight': weight, 'timestamp': timestamp, 'source': source}
    )

    # Handle rollbacks if needed
    for confirmation in confirmations:
        if confirmation.get('rollback_required'):
            perform_rollback(user_id, confirmation['provisional_id'])

    # Continue with normal processing...
```

## Implementation Steps

### Phase 1: Core Framework (Week 1)
1. Create `ProvisionalManager` class
2. Implement provisional state tracking
3. Add database schema for provisional measurements
4. Create status enum and data structures

### Phase 2: Decision Logic (Week 2)
1. Implement `ProvisionalDecisionMaker`
2. Define provisional acceptance criteria
3. Create confirmation scoring algorithm
4. Add expiration handling

### Phase 3: Confirmation System (Week 3)
1. Implement confirmation checking
2. Add rollback mechanism
3. Create state snapshot/restore functionality
4. Handle expired provisional measurements

### Phase 4: Integration (Week 4)
1. Integrate with main processor
2. Update UI to show provisional status
3. Add monitoring and alerts
4. Create provisional reports

## Testing Strategy

### Unit Tests
```python
def test_provisional_after_gap():
    """Test provisional acceptance after long gap."""
    mgr = ProvisionalManager(mock_db)
    result = mgr.accept_provisional(
        'user1',
        {'weight': 140, 'timestamp': datetime.now()},
        'long_gap_uncertainty'
    )
    assert result['provisional']
    assert result['expires_at'] > datetime.now()

def test_confirmation_by_subsequent():
    """Test confirmation by subsequent measurements."""
    mgr = ProvisionalManager(mock_db)
    # Accept provisionally
    provisional = mgr.accept_provisional(...)

    # Add confirming measurements
    for weight in [138, 139, 141]:
        confirmations = mgr.check_confirmations('user1', {'weight': weight})

    # Should be confirmed
    assert confirmations[-1]['action'] == 'confirmed'
```

### Integration Tests
1. Test full provisional flow with real data
2. Test rollback on rejection
3. Test expiration handling
4. Test multiple provisional measurements

## Configuration

```toml
[provisional_acceptance]
enabled = true
confirmation_window_hours = 72
min_confirming_measurements = 3
confirmation_threshold = 0.7

[provisional_acceptance.triggers]
gap_threshold_days = 30
quality_score_min = 0.3
quality_score_max = 0.6
change_threshold = 0.15
```

## Success Metrics

1. **Reduction in false rejections**: <5% valid measurements rejected
2. **Confirmation accuracy**: >85% provisional measurements correctly classified
3. **Rollback frequency**: <10% of provisional acceptances need rollback
4. **User experience**: No visible delays from provisional processing

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| State corruption from rollback | High | Comprehensive state snapshots, testing |
| Cascade effects from rejection | Medium | Limit provisional chain length |
| Memory overhead | Low | Expire old provisional records |
| User confusion | Medium | Clear UI indicators, documentation |

## Future Enhancements

1. Machine learning for provisional decision making
2. User-specific provisional thresholds
3. Automatic pattern learning from confirmations
4. Integration with manual review queue
5. Provisional confidence decay over time