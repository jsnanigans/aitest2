# Plan: CSV Data Preprocessor Script

## Summary
Create a preprocessing script that transforms raw CSV weight data into an optimized format for the main weight stream processor, with user grouping, chronological sorting, and configurable filtering.

## Context
- **Source**: User request for data preprocessing functionality
- **Input Format**: CSV files with columns: user_id, effectiveDateTime, source_type, weight, unit
- **Output Format**: Optimized CSV with same structure but preprocessed data
- **Integration**: Feeds into main.py weight stream processor

### Assumptions
- Input CSV follows standard format from test_sample.csv
- All weights are in kg (or convertible to kg)
- effectiveDateTime is parseable (various formats accepted)
- User IDs are consistent within the file
- Source filtering should be case-insensitive

### Constraints
- Must maintain CSV format compatibility with main.py
- Should handle large files efficiently (streaming if needed)
- Must preserve all original columns for downstream processing
- Should be runnable as standalone script

## Requirements

### Functional
1. Parse input CSV file (e.g., `2025-09-05.csv`)
2. Group all measurements by user_id
3. Sort measurements within each user group by effectiveDateTime
4. Filter by configurable source types (ignore list)
5. Filter by configurable time range (min/max datetime)
6. Output optimized CSV (e.g., `2025-09-05_optimized.csv`)
7. Provide summary statistics (users processed, rows filtered, etc.)

### Non-functional
- **Performance**: Handle files with 100k+ rows efficiently
- **Memory**: Stream processing for large files (avoid loading all data at once)
- **Reliability**: Handle malformed dates, missing values gracefully
- **Usability**: Clear configuration section at script top
- **Compatibility**: Python 3.11+ with minimal dependencies

## Alternatives

### Option A: In-Memory Processing
**Approach**: Load entire CSV into memory, process with pandas/dictionaries
- **Pros**: 
  - Simple implementation
  - Fast for small-medium files
  - Easy sorting and grouping with pandas
- **Cons**: 
  - Memory issues with large files
  - Not scalable for production use
- **Risks**: OOM errors on large datasets

### Option B: Streaming with Buffered Groups
**Approach**: Stream file twice - first pass to identify users, second to process
- **Pros**: 
  - Memory efficient
  - Handles arbitrarily large files
  - Can process users as completed
- **Cons**: 
  - Two-pass algorithm slower
  - More complex implementation
- **Risks**: Temporary file management

### Option C: Hybrid Approach
**Approach**: Stream input, buffer per-user data, write completed users immediately
- **Pros**: 
  - Balance of memory and speed
  - Single pass through input
  - Configurable memory limits
- **Cons**: 
  - Moderate complexity
  - Need to handle user buffer overflow
- **Risks**: Edge cases with very large single-user datasets

## Recommendation
**Option C: Hybrid Approach** - Best balance of performance and memory efficiency. Buffer individual users in memory (reasonable assumption that single user won't have millions of records), write each user's sorted data immediately after processing.

**Rationale**: 
- Most scalable for production use
- Single pass efficiency
- Graceful handling of edge cases
- Aligns with streaming philosophy of main processor

## High-Level Design

### Architecture Flow
```
Input CSV → Parse Row → Filter (source/time) → Buffer by User → 
Sort User Data → Write to Output → Next User → Summary Stats
```

### Data Model
- **Input Row**: Dict with user_id, effectiveDateTime, source_type, weight, unit
- **User Buffer**: Dict[user_id] → List[Row]
- **Config**: Dataclass with ignore_sources, min_time, max_time, etc.

### Affected Modules/Files
- **New File**: `scripts/preprocess_csv.py` (standalone script)
- **Integration**: Output feeds into main.py via config.toml update
- **No changes** to existing processor modules

### Configuration Structure
```python
# At top of script
CONFIG = {
    'ignore_sources': ['test-source', 'manual-entry'],  # Sources to filter out
    'min_datetime': '2020-01-01 00:00:00',  # Earliest date to include
    'max_datetime': '2025-12-31 23:59:59',  # Latest date to include
    'max_buffer_size': 10000,  # Max records per user in memory
    'output_suffix': '_optimized',  # Suffix for output filename
    'verbose': True,  # Print progress updates
}
```

## Implementation Plan (No Code)

### Step 1: Script Setup and Configuration
- Create `scripts/preprocess_csv.py`
- Define CONFIG dictionary at top
- Add argparse for CLI arguments (input file, optional output)
- Import required modules (csv, datetime, pathlib, sys)

### Step 2: Date/Time Parsing Utilities
- Create flexible datetime parser function
- Handle multiple formats (ISO, various separators)
- Return None for unparseable dates (with warning)
- Cache parsed dates for efficiency

### Step 3: Source Filtering Logic
- Normalize source strings (lowercase, strip)
- Check against ignore list
- Optional: regex pattern matching for sources

### Step 4: Main Processing Loop
- Open input CSV with csv.DictReader
- Initialize user buffers dictionary
- For each row:
  - Parse and validate datetime
  - Check source filter
  - Check time range filter
  - Add to user buffer
  - If buffer full, flush user to output

### Step 5: User Data Processing
- Sort user's measurements by effectiveDateTime
- Remove exact duplicates (same time, weight, source)
- Optional: merge very close timestamps (< 1 minute apart)

### Step 6: Output Writing
- Create output filename from input (add suffix)
- Write CSV header
- Stream sorted user data to output
- Maintain original column order

### Step 7: Statistics and Reporting
- Track: total rows, filtered rows, users processed
- Source type distribution
- Date range of output data
- Print summary to console

### Step 8: Error Handling
- Graceful handling of malformed rows
- Log errors to separate file
- Continue processing on non-fatal errors

## Validation & Rollout

### Test Strategy
1. **Unit Tests**:
   - Date parsing with various formats
   - Source filtering logic
   - Time range filtering
   - Sorting within user groups

2. **Integration Tests**:
   - Small test file (10 users, 100 rows)
   - Large test file (1000 users, 100k rows)
   - Edge cases (single user, no valid data)
   - Malformed data handling

3. **End-to-end Tests**:
   - Preprocess → main.py pipeline
   - Compare results with/without preprocessing
   - Performance benchmarks

### Manual QA Checklist
- [ ] Verify output CSV format matches input
- [ ] Check user grouping is correct
- [ ] Confirm chronological sorting per user
- [ ] Test source filtering works
- [ ] Test time range filtering works
- [ ] Verify statistics are accurate
- [ ] Test with real 2025-09-05.csv file

### Rollout Plan
1. **Phase 1**: Implement core functionality
2. **Phase 2**: Add filtering and configuration
3. **Phase 3**: Add statistics and error handling
4. **Phase 4**: Performance optimization if needed
5. **Phase 5**: Integration with main.py workflow

## Risks & Mitigations

### Risk 1: Memory Usage with Large Files
- **Mitigation**: Implement buffer size limits per user
- **Mitigation**: Add streaming mode for huge files
- **Monitor**: Memory usage during processing

### Risk 2: Date Format Variations
- **Mitigation**: Flexible parser with multiple format attempts
- **Mitigation**: Log unparseable dates for review
- **Fallback**: Skip rows with invalid dates

### Risk 3: Performance on Large Datasets
- **Mitigation**: Profile and optimize hot paths
- **Mitigation**: Consider multiprocessing for user groups
- **Option**: Implement progress bar for long runs

## Acceptance Criteria

1. ✓ Script processes test_sample.csv successfully
2. ✓ Output CSV maintains all original columns
3. ✓ Measurements grouped by user_id
4. ✓ Each user's data sorted by effectiveDateTime
5. ✓ Source filtering removes configured sources
6. ✓ Time filtering respects min/max boundaries
7. ✓ Summary statistics printed to console
8. ✓ Output filename follows pattern: `input_optimized.csv`
9. ✓ Script handles malformed data gracefully
10. ✓ Compatible with main.py processor

## Out of Scope

- Database integration (CSV only)
- Weight unit conversion (assumed kg)
- Data validation beyond basic parsing
- Outlier detection (handled by main processor)
- User deduplication across files
- REST API or web interface
- Real-time streaming processing
- Cloud storage integration

## Open Questions

1. **Source Filtering**: Should we support regex patterns or just exact matches?
2. **Duplicate Handling**: How to handle exact duplicate measurements (same user, time, weight)?
3. **Memory Limits**: What's acceptable memory usage for buffering?
4. **Output Location**: Same directory as input or configurable output path?
5. **Logging**: Should we create a detailed log file or just console output?
6. **Performance**: Is multiprocessing needed for very large files?
7. **Validation**: Should we validate weight values (e.g., reasonable ranges)?

## Review Cycle

### Self-Review Notes
- ✓ Plan covers all stated requirements
- ✓ Hybrid approach balances performance and memory
- ✓ Configuration is user-friendly and at script top
- ✓ Integration with main.py considered
- ✓ Error handling addressed
- ✓ Test strategy comprehensive

### Revisions
- Added buffer overflow handling consideration
- Clarified streaming approach for large files
- Added duplicate detection within user groups
- Specified Python 3.11+ requirement
- Added progress reporting for long runs

### Council Input Needed

**Butler Lampson** would say: "Keep it simple - a straightforward CSV processor that does one thing well. The hybrid approach is good but watch for complexity creep."

**Grace Hopper** would advise: "Make sure the data formats are well-documented. Future users need to understand the preprocessing transformations."

**Don Norman** would note: "The configuration at the top is good for usability. Consider adding example values and clear documentation for each setting."

**Ward Cunningham** would suggest: "Document the data transformations clearly. What exactly changes between input and output?"

### Next Steps
1. Review and approve plan
2. Implement core CSV processing
3. Add filtering capabilities  
4. Integrate with main.py workflow
5. Test with production data