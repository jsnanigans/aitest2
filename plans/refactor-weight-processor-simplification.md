# Plan: Weight Processor Simplification Refactoring

## Summary
Implement Architecture Council recommendations to reduce codebase by 70%, improve performance 10x, and dramatically simplify the weight processing pipeline while preserving core Kalman filtering functionality.

## Context
- Source: Architecture Council Review (docs/COUNCIL_ARCHITECTURE_REVIEW.md)
- Current state: ~5000 lines of accumulated complexity
- Target state: ~1500 lines of focused, clear code
- Assumptions:
  - Core Kalman filtering logic is sound and must be preserved
  - Source-specific noise adaptation is valuable
  - Many features were exploratory and can be removed
- Constraints:
  - Must maintain backward compatibility with CSV input format
  - Must preserve accuracy of weight filtering
  - Cannot break existing data processing for active users

## Requirements

### Functional
- Maintain Kalman filtering with adaptive noise
- Preserve physiological validation
- Keep source-specific processing
- Support CSV input/output
- Generate visualizations

### Non-functional
- 10x performance improvement (30ms → 3ms per measurement)
- 70% code reduction
- Improved maintainability (single-purpose functions)
- Clear architectural boundaries
- Comprehensive test coverage

## Alternatives

### Option A: Incremental Refactoring
- Approach: Gradual cleanup while maintaining all features
- Pros: Lower risk, can be done in production
- Cons: Slower, may not achieve full simplification, technical debt remains

### Option B: Parallel Rewrite
- Approach: Build simplified version alongside current
- Pros: No disruption to current system, clean slate
- Cons: Duplicate effort, migration complexity, longer timeline

### Option C: Focused Refactoring (Recommended)
- Approach: Aggressive simplification in phases with testing
- Pros: Fast, achieves maximum simplification, clear wins
- Cons: Higher risk, requires careful testing

## Recommendation
Option C - Focused Refactoring. The codebase has clear dead code and the core algorithm is well-understood. A focused effort will yield the best results.

## High-Level Design

### Simplified Architecture
```
main.py
├── processor.py (single processing pipeline)
├── kalman.py (core filter only)
├── database.py (simple state store)
├── validation.py (merged validation + quality)
└── visualization.py (essential plots only)
```

### Data Flow
```
CSV Input → process_measurement() → Database → Results
              ├── validate()           ↓
              ├── kalman_update()    State
              └── save_state()
```

### Key Consolidations
- Merge validation.py + quality.py → validation.py
- Merge reprocessor.py → processor.py  
- Eliminate models.py (move constants to modules)
- Remove all backup/broken files
- Delete analysis/debug scripts

## Implementation Plan (No Code)

### Phase 1: Dead Code Removal (Day 1)
1. **Delete abandoned files**
   - Remove visualization_backup.py, visualization_broken.py
   - Delete scripts/analysis/* (keep only essential)
   - Remove scripts/debug/*
   - Delete DynamicResetManager class
   - Remove snapshot/rollback code from database.py

2. **Clean up test suite**
   - Identify redundant tests (target: 80 → 6 files)
   - Consolidate into: test_processor.py, test_kalman.py, test_validation.py, test_database.py, test_visualization.py, test_integration.py
   - Remove all investigation/debug tests

3. **Simplify configuration**
   - Remove unused parameters from config.toml
   - Move test_users to separate file
   - Hard-code physiological safety limits
   - Move source profiles to code constants

### Phase 2: Pipeline Flattening (Days 2-3)
1. **Consolidate processing functions**
   - Merge process_weight_enhanced + WeightProcessor.process_weight + _process_weight_internal
   - Create single process_measurement() function
   - Move orchestration logic to main.py

2. **Simplify state management**
   - Remove daily_measurements cache
   - Eliminate user_debug_logs accumulation
   - Stream results instead of accumulating
   - Single source of truth: database

3. **Merge validation modules**
   - Combine validation.py and quality.py
   - Simplify to essential validation functions
   - Remove class wrappers where unnecessary

### Phase 3: Optimization (Days 4-5)
1. **Performance improvements**
   - Cache user heights in database
   - Pre-calculate thresholds per session
   - Remove numpy ↔ list conversions
   - Eliminate dictionary recreation

2. **Module consolidation**
   - Inline simple functions
   - Remove unnecessary abstractions
   - Consolidate constants
   - Eliminate circular dependencies

3. **Simplify Kalman implementation**
   - Remove redundant state tracking
   - Optimize matrix operations
   - Simplify confidence calculations

### Phase 4: Hardening (Day 6)
1. **Add safety features**
   - Input validation for CSV
   - Hard-coded physiological limits
   - Error boundaries
   - Graceful degradation

2. **Improve observability**
   - Structured logging (not verbose debug)
   - Performance metrics
   - Error tracking
   - Health checks

## Validation & Rollout

### Test Strategy
1. **Before refactoring**: Create golden dataset of current outputs
2. **Unit tests**: Each module independently
3. **Integration tests**: Full pipeline with real data
4. **Regression tests**: Compare against golden dataset
5. **Performance tests**: Verify 10x improvement
6. **Load tests**: Ensure memory efficiency

### Manual QA Checklist
- [ ] Process sample CSV successfully
- [ ] Kalman filtering produces same results
- [ ] Visualizations generate correctly
- [ ] Source-specific handling works
- [ ] Gap detection triggers appropriately
- [ ] Performance meets 3ms target

### Rollout Plan
1. **Phase 1**: Deploy to test environment
2. **Phase 2**: Process historical data, compare outputs
3. **Phase 3**: Parallel run with production
4. **Phase 4**: Gradual migration of users
5. **Phase 5**: Full cutover and cleanup

## Risks & Mitigations

### Risk 1: Breaking existing functionality
- **Mitigation**: Comprehensive test suite before changes
- **Mitigation**: Golden dataset for regression testing

### Risk 2: Performance regression
- **Mitigation**: Benchmark before/after each phase
- **Mitigation**: Profile code to identify bottlenecks

### Risk 3: Lost edge case handling
- **Mitigation**: Document each validation before removal
- **Mitigation**: Keep git history for reference

### Risk 4: Data corruption
- **Mitigation**: Database backups before changes
- **Mitigation**: Parallel run before cutover

## Acceptance Criteria

### Quantitative
- [ ] Code size reduced by >60% (target: 70%)
- [ ] Performance <5ms per measurement (target: 3ms)
- [ ] Test execution time <10 seconds
- [ ] Memory usage reduced by >40%
- [ ] All regression tests pass

### Qualitative
- [ ] Single-purpose functions throughout
- [ ] Clear module boundaries
- [ ] No circular dependencies
- [ ] Comprehensive documentation
- [ ] New developer can understand in <2 hours

## Out of Scope
- Changing CSV format
- Adding new features
- Database persistence (stays in-memory)
- Authentication/authorization
- Distributed processing

## Open Questions
1. Should we preserve the daily cleanup feature or simplify to streaming only?
2. Is the height data CSV critical or can we simplify BMI detection?
3. Should visualization be part of core or separate utility?
4. Do we need to maintain backward compatibility with old state format?

## Review Cycle

### Self-Review Notes
- Plan aligns with Council recommendations
- Phases are logical and buildable
- Risk mitigation is comprehensive
- Timeline is aggressive but achievable

### Revisions
- Added specific file lists for deletion
- Clarified consolidation strategy
- Added performance benchmarking requirements
- Specified regression testing approach

## Success Metrics
- **Week 1**: All dead code removed, tests consolidated
- **Week 2**: Pipeline flattened, performance improved
- **Week 3**: Full deployment, monitoring stability
- **Month 1**: 0 regression issues, improved developer velocity

## Next Steps
1. Get stakeholder approval
2. Create golden dataset for regression testing
3. Set up performance benchmarking
4. Begin Phase 1: Dead code removal
5. Daily progress updates during implementation