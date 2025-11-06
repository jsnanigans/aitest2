# Test Fixtures

This directory contains test data fixtures for manual testing and development of the weight processor.

## Files

### Replay Testing
- **`replay_scenario.json`**: Test data demonstrating buffered replay triggers with time gaps
  - 4 measurements spanning multiple days
  - Designed to trigger time_gap replay
  - Referenced in `scripts/manual_testing/test_replay_trigger.py`

### Processing Testing
- **`process_request.json`**: Sample request payload for processing measurements
- **`process_local.json`**: Local invocation test data for batch processing

### Cleanup Testing
- **`cleanup_request.json`**: Sample cleanup request payload
- **`cleanup_local.json`**: Local invocation test data for cleanup operations

### Response Examples
- **`lambda_response_example.json`**: Example Lambda response showing:
  - Multiple replay triggers (time_window)
  - Quality scoring results
  - Acceptance/rejection decisions
  - Replay metadata with timing and buffer information

## Usage

These fixtures are intended for:
1. Manual testing via scripts in `scripts/manual_testing/`
2. Local Lambda invocation testing
3. Development and debugging
4. Documentation and examples

## Related Documentation

- See `../../BUFFERED_REPLAY.md` for buffered replay system design
- See `../../scripts/manual_testing/` for manual test scripts
- See `../integration/test_buffered_replay.py` for automated integration tests
