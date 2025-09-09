import pytest
from datetime import datetime, timedelta
from src.filters.enhanced_validation_gate import EnhancedValidationGate


class TestEnhancedValidationGate:
    
    def test_duplicate_detection_bug_01672f42(self):
        """
        Test case for user 01672f42-568b-4d49-abbc-eee60d87ccb2
        Bug: should_deduplicate doesn't reset today_readings when date changes,
        causing readings from different days to be incorrectly marked as duplicates
        """
        gate = EnhancedValidationGate({
            'duplicate_threshold_kg': 0.5,
            'max_deviation_pct': 0.30,  # Allow 30% deviation
            'same_day_threshold_kg': 10.0
        })
        
        user_id = "01672f42-568b-4d49-abbc-eee60d87ccb2"
        
        # Simulate the main.py flow: deduplicate check happens BEFORE validation
        
        # Day 1: Add reading that will be validated
        day1_reading = {
            'date': datetime(2025, 5, 27, 17, 32, 19),
            'weight': 106.1,
            'source': 'api'
        }
        
        # First check duplicate (should be false - no context yet)
        is_dup = gate.should_deduplicate(user_id, day1_reading)
        assert is_dup == False, "First reading should not be duplicate"
        
        # Then validate (this adds to context)
        if not is_dup:
            gate.validate_reading(user_id, day1_reading)
        
        # Day 1: Second reading, same weight - should be duplicate
        day1_reading2 = {
            'date': datetime(2025, 5, 27, 17, 33, 0),
            'weight': 106.1,  # Same weight
            'source': 'api'
        }
        
        is_dup2 = gate.should_deduplicate(user_id, day1_reading2)
        assert is_dup2 == True, "Same weight on same day should be duplicate"
        
        # Day 2: Different day, different weight - BUG: marked as duplicate
        day2_reading = {
            'date': datetime(2025, 5, 30, 13, 52, 56),
            'weight': 107.3,  # Different weight, different day
            'source': 'api'
        }
        
        # This is the BUG: should_deduplicate doesn't check if day changed
        # So it compares 107.3 against day1's readings (106.1)
        # Since 107.3 - 106.1 = 1.2kg > 0.5kg threshold, it SHOULD pass
        # But if context.today_readings still has old data, it might fail
        is_dup3 = gate.should_deduplicate(user_id, day2_reading)
        
        # The bug would cause this to fail if today_readings isn't cleared
        # In the actual bug, the context.current_day stays on old date
        assert is_dup3 == False, f"Different day should reset duplicates, but got duplicate={is_dup3}"
        
        # Verify the context state to understand the bug
        context = gate.user_contexts[user_id]
        reading_date = day2_reading['date'].date()
        
        # The bug is that should_deduplicate doesn't update current_day
        # So context.current_day might still be May 27, not May 30
        # This test exposes that issue
    
    def test_duplicate_detection_day_boundary_bug(self):
        """
        Specific test for the day boundary bug in should_deduplicate.
        The bug: should_deduplicate doesn't reset today_readings when crossing day boundary.
        """
        gate = EnhancedValidationGate({'duplicate_threshold_kg': 0.5})
        user_id = "test_user"
        
        # Day 1: Add a reading via validation (populates context)
        day1_reading = {
            'date': datetime(2025, 5, 28, 10, 0, 0),
            'weight': 107.7,
            'source': 'test'
        }
        gate.validate_reading(user_id, day1_reading)
        
        # Verify context state
        context = gate.user_contexts[user_id]
        assert context.current_day == datetime(2025, 5, 28).date()
        assert context.today_readings == [107.7]
        
        # Day 2: Check duplicate for new day - this reveals the bug
        day2_reading = {
            'date': datetime(2025, 5, 30, 10, 0, 0),
            'weight': 107.3,  # Within 0.5kg of 107.7
            'source': 'test'
        }
        
        # FIXED: should_deduplicate now checks/updates current_day
        is_dup = gate.should_deduplicate(user_id, day2_reading)
        
        # This should now be False (different day resets duplicates)
        assert is_dup == False, "Different day should not be marked as duplicate"
        
        # After the check, context.current_day should be updated
        assert context.current_day == datetime(2025, 5, 30).date(), "current_day should be updated by should_deduplicate"
        assert context.today_readings == [], "today_readings should be reset for new day"
    
    def test_today_readings_reset_on_new_day(self):
        """Test that today_readings is properly reset when date changes"""
        gate = EnhancedValidationGate({
            'duplicate_threshold_kg': 0.5
        })
        
        user_id = "test_user"
        
        # Add readings for day 1
        day1_reading = {
            'date': datetime(2025, 1, 1, 10, 0, 0),
            'weight': 70.0,
            'source': 'test'
        }
        
        gate.validate_reading(user_id, day1_reading)
        
        # Check context has reading for day 1
        context = gate.user_contexts[user_id]
        assert len(context.today_readings) == 1
        assert context.current_day == datetime(2025, 1, 1).date()
        
        # Add reading for day 2
        day2_reading = {
            'date': datetime(2025, 1, 2, 10, 0, 0),
            'weight': 70.5,
            'source': 'test'
        }
        
        gate.validate_reading(user_id, day2_reading)
        
        # Today readings should be reset
        assert len(context.today_readings) == 1  # Only new reading
        assert context.today_readings[0] == 70.5
        assert context.current_day == datetime(2025, 1, 2).date()
    
    def test_multiple_users_isolation(self):
        """Test that duplicate detection is isolated per user"""
        gate = EnhancedValidationGate({
            'duplicate_threshold_kg': 0.5
        })
        
        user1 = "user1"
        user2 = "user2"
        
        # Same weight, same day for both users
        reading = {
            'date': datetime(2025, 1, 1, 10, 0, 0),
            'weight': 70.0,
            'source': 'test'
        }
        
        # Both should pass (not duplicates across users)
        assert gate.should_deduplicate(user1, reading) == False
        gate.validate_reading(user1, reading)
        
        assert gate.should_deduplicate(user2, reading) == False
        gate.validate_reading(user2, reading)
        
        # Second reading for user1 with same weight - should be duplicate
        reading2 = {
            'date': datetime(2025, 1, 1, 11, 0, 0),
            'weight': 70.2,  # Within 0.5kg
            'source': 'test'
        }
        
        assert gate.should_deduplicate(user1, reading2) == True
        # User2 also has 70.0, so 70.2 is also duplicate for user2 (within 0.5kg)
        assert gate.should_deduplicate(user2, reading2) == True
        
        # But a different weight for user2 that user1 doesn't have
        reading3 = {
            'date': datetime(2025, 1, 1, 12, 0, 0),
            'weight': 75.0,  # Not within 0.5kg of either user's readings
            'source': 'test'
        }
        
        assert gate.should_deduplicate(user1, reading3) == False  # 75.0 is >0.5kg from 70.0/70.2
        assert gate.should_deduplicate(user2, reading3) == False  # 75.0 is >0.5kg from 70.0/70.2
    
    def test_validation_checks_order(self):
        """Test that validation checks happen in correct order"""
        gate = EnhancedValidationGate({
            'max_deviation_pct': 0.15,
            'rapid_change_threshold_pct': 0.10,
            'rapid_change_hours': 24,
            'future_date_tolerance_days': 1
        })
        
        user_id = "test_user"
        
        # Establish baseline
        for i in range(5):
            reading = {
                'date': datetime(2025, 1, 1 + i, 10, 0, 0),
                'weight': 70.0 + i * 0.1,
                'source': 'test'
            }
            gate.validate_reading(user_id, reading)
        
        # Test future date rejection
        future_reading = {
            'date': datetime.now() + timedelta(days=7),
            'weight': 70.0,
            'source': 'test'
        }
        is_valid, reason = gate.validate_reading(user_id, future_reading)
        assert is_valid == False
        assert reason == "Future date"
        
        # Test extreme deviation
        extreme_reading = {
            'date': datetime(2025, 1, 6, 10, 0, 0),
            'weight': 85.0,  # >15% deviation
            'source': 'test'
        }
        is_valid, reason = gate.validate_reading(user_id, extreme_reading)
        assert is_valid == False
        assert "Extreme deviation" in reason
    
    def test_baseline_establishment(self):
        """Test that baseline is properly established after enough readings"""
        gate = EnhancedValidationGate({})
        user_id = "test_user"
        
        # Add readings
        weights = [70.0, 70.5, 69.8, 70.2, 70.1]
        for i, weight in enumerate(weights):
            reading = {
                'date': datetime(2025, 1, 1 + i, 10, 0, 0),
                'weight': weight,
                'source': 'test'
            }
            gate.validate_reading(user_id, reading)
        
        # Check baseline was established
        context = gate.user_contexts[user_id]
        assert context.baseline_weight is not None
        assert 69.5 <= context.baseline_weight <= 70.5  # Should be near median


if __name__ == "__main__":
    pytest.main([__file__, "-v"])