# Plan: Add Database Dump CSV Export

## Summary
Add functionality to export the processor state database entries as CSV format, providing visibility into the stored Kalman filter states and user processing history.

## Context
- Source: User description
- Assumptions:
  - Users want to inspect the internal state database contents
  - CSV format is preferred for easy analysis in spreadsheets/data tools
  - The dump should include all relevant state information in a readable format
- Constraints:
  - Must handle serialized numpy arrays and datetime objects properly
  - Should not interfere with normal processing flow
  - CSV output should be human-readable and analysis-friendly

## Requirements

### Functional
- Export all user states from ProcessorStateDB to CSV format
- Include key state information: user_id, last_timestamp, last_weight, last_trend, kalman parameters
- Handle numpy arrays and complex objects gracefully
- Save CSV file alongside other output files with timestamp
- Optionally include in main processing output or as separate command

### Non-functional
- Performance: Should not significantly slow down processing
- Usability: CSV should be easily importable into Excel/pandas
- Maintainability: Clean integration with existing codebase

## Alternatives

### Option A: Inline Database Dump During Processing
- Approach: Add CSV export at the end of main.py processing
- Pros:
  - Automatic with every run
  - Captures final state after all processing
  - No additional commands needed
- Cons:
  - Always generates file even if not needed
  - Adds processing time to every run
- Risks: Minimal

### Option B: Separate CLI Command
- Approach: Add new command-line option `--dump-db` to export database
- Pros:
  - On-demand only when needed
  - Can be run independently of processing
  - More flexible for debugging
- Cons:
  - Requires separate invocation
  - May miss transient states during processing
- Risks: None

### Option C: Both Inline and CLI Option
- Approach: Add to processing output by default with option to disable, plus standalone command
- Pros:
  - Maximum flexibility
  - Always available in output
  - Can be disabled if not wanted
- Cons:
  - Slightly more complex implementation
- Risks: None

## Recommendation
**Option C: Both Inline and CLI Option**

Rationale: Provides the best of both worlds - automatic capture of final states after processing, with the flexibility to dump the database at any time. The overhead of CSV generation is minimal compared to processing time.

## High-Level Design

### Architecture
```
ProcessorStateDB
    ↓
export_to_csv() method
    ↓
CSV Writer (handles serialization)
    ↓
output/db_dump_<timestamp>.csv
```

### Components to Modify
1. `src/database.py`: Add `export_to_csv()` method to ProcessorStateDB
2. `main.py`: Add CSV export after processing and CLI option
3. Config: Add option to control database dump behavior

### CSV Structure
```csv
user_id,last_timestamp,last_weight,last_trend,has_kalman_params,process_noise,measurement_noise,state_reset_count
user123,2024-01-15T10:30:00,75.2,0.05,true,0.01,1.0,2
user456,2024-01-15T09:15:00,82.1,-0.02,true,0.01,1.0,0
...
```

## Implementation Plan (No Code)

### Step 1: Add CSV Export Method to ProcessorStateDB
- Location: `src/database.py`
- Add `export_to_csv(filepath: str)` method
- Iterate through all states in self.states
- Extract and flatten key fields from each user's state
- Handle numpy array conversion (extract scalar values)
- Handle datetime serialization
- Write CSV with appropriate headers

### Step 2: Integrate into Main Processing Flow
- Location: `main.py`, after line ~600 (after debug reports)
- Call db.export_to_csv() with timestamped filename
- Save to output directory alongside other results
- Add console output indicating CSV export

### Step 3: Add CLI Option for Standalone Export
- Location: `main.py` argument parser section
- Add `--dump-db` flag
- Add `--db-output` option for custom output path
- If flag present, create DB instance, export, and exit

### Step 4: Add Configuration Option
- Location: `config.toml`
- Add `[data]` section option: `export_database = true`
- Allow disabling with `--no-db-dump` CLI flag

### Step 5: Handle Edge Cases
- Empty database (no users processed)
- Partially initialized states
- States without Kalman parameters
- Very large databases (chunked writing if needed)

## Validation & Rollout

### Test Strategy
1. Unit test for export_to_csv method
2. Test with empty database
3. Test with single user
4. Test with multiple users
5. Test CSV import in pandas/Excel
6. Test standalone CLI command
7. Test with very large state database

### Manual QA Checklist
- [ ] CSV file created in output directory
- [ ] All processed users appear in CSV
- [ ] Timestamps are readable
- [ ] Weight and trend values are correct
- [ ] CSV imports cleanly into Excel
- [ ] CSV loads properly in pandas
- [ ] Standalone --dump-db command works
- [ ] --no-db-dump flag disables export

### Rollout Plan
1. Implement and test locally
2. Run on test dataset to verify output
3. Document CSV format in README
4. Deploy to production

## Risks & Mitigations

### Risk 1: Large Memory Usage
- **Risk**: Very large databases might cause memory issues during export
- **Mitigation**: Implement streaming CSV writer if needed
- **Monitoring**: Track memory usage during export

### Risk 2: Sensitive Data Exposure
- **Risk**: CSV might contain sensitive user information
- **Mitigation**: Ensure proper file permissions, consider anonymization option
- **Monitoring**: Review data classification requirements

## Acceptance Criteria
- [ ] Database states are exported to CSV file after processing
- [ ] CSV contains: user_id, last_timestamp, last_weight, last_trend, kalman params presence
- [ ] CSV file is saved with timestamp in output directory
- [ ] Console shows "Database dump saved to..." message
- [ ] --dump-db flag works independently
- [ ] --no-db-dump flag disables automatic export
- [ ] CSV is importable in Excel and pandas

## Out of Scope
- Exporting historical state snapshots
- Exporting full covariance matrices (too complex for CSV)
- Real-time database monitoring
- Database import from CSV
- Encryption or compression of CSV output

## Open Questions
1. Should we include more Kalman parameters (process noise, measurement noise values)?
2. Should we add a summary row with aggregate statistics?
3. Should the CSV include user processing statistics (acceptance rate, rejection count)?
4. Do we need different export formats (JSON, SQLite) in the future?

## Review Cycle
- Self-review notes:
  - Considered including full state vectors but decided against for simplicity
  - CSV format chosen over JSON for better spreadsheet compatibility
  - Timestamp format will use ISO 8601 for consistency
  - File naming convention matches existing output files pattern