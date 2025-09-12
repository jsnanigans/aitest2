# Plan: Clean Up Src Directory

## Summary
Consolidate and clean up the src directory to achieve a minimal three-file architecture (processor, database, visualization) while removing all unused/legacy code. The goal is to have a clean, maintainable codebase with logical separation of concerns.

## Context
- Source: User request for src directory cleanup
- Current state: 10 files in src/, many appear to be legacy/backup versions
- Main.py currently imports from: processor, processor_database, reprocessor, visualization
- Tests import from various src files including enhanced/dynamic variants
- Assumptions:
  - The latest working implementation is in processor.py (with process_weight_enhanced)
  - processor_database.py handles all state persistence
  - visualization.py handles all visualization logic
  - Other files are likely iterations/experiments that can be consolidated

## Requirements
### Functional
- Maintain all current functionality (weight processing, state management, visualization)
- Preserve test compatibility (update imports as needed)
- Keep the stateless architecture pattern intact
- Maintain BMI validation and data quality features

### Non-functional
- Maximum 3-4 core files in src/
- Clear separation of concerns
- No code duplication
- Clean, logical naming

## Alternatives

### Option A: Strict Three-File Architecture
- Approach: Merge everything into exactly 3 files (processor.py, database.py, visualization.py)
- Pros:
  - Absolute minimal file count
  - Very clear structure
  - Easy to navigate
- Cons:
  - May result in very large files
  - Could mix unrelated concerns
  - Harder to test individual components
- Risks: Loss of modularity, difficult maintenance if files become too large

### Option B: Core Three + Essential Utilities
- Approach: Keep 3 core files plus 1-2 utility files for specific concerns (e.g., reprocessor.py for batch operations)
- Pros:
  - Clean separation while maintaining modularity
  - Easier to test specific features
  - More maintainable file sizes
- Cons:
  - Slightly more files than absolute minimum
  - Need to decide what qualifies as "essential"
- Risks: Scope creep leading to more utility files over time

### Option C: Feature-Based Consolidation
- Approach: Organize by feature domains (processing, persistence, visualization, validation)
- Pros:
  - Logical grouping of related functionality
  - Good balance of file count and organization
  - Natural boundaries for testing
- Cons:
  - May result in 4-5 files instead of 3
  - Some features span multiple domains
- Risks: Unclear boundaries between domains

## Recommendation
**Option B: Core Three + Essential Utilities**

Rationale:
- Maintains the desired three-file core structure
- Allows keeping reprocessor.py as it serves a distinct batch processing purpose
- Achieves 90% reduction in file count while maintaining code clarity
- Easier migration path with less risk of breaking changes

## High-Level Design

### Target Structure
```
src/
├── processor.py         # All weight processing logic (stateless)
├── database.py          # State persistence (renamed from processor_database.py)
├── visualization.py     # All visualization logic
└── reprocessor.py       # Batch/daily cleanup operations (kept as utility)
```

### Consolidation Map
1. **processor.py** (merge into existing):
   - Keep current process_weight_enhanced, DataQualityPreprocessor
   - Merge BMIValidator class from bmi_validator.py
   - Merge ThresholdCalculator from threshold_calculator.py
   - Merge DynamicResetManager from dynamic_reset_manager.py
   - Keep all WeightProcessor static methods

2. **database.py** (rename from processor_database.py):
   - Keep all existing ProcessorStateDB, ProcessorDatabase classes
   - No changes needed, just rename for consistency

3. **visualization.py**:
   - Already complete, no changes needed

4. **reprocessor.py**:
   - Keep as-is for batch processing operations
   - Already well-integrated with main.py

### Files to Remove
- processor_backup.py (old backup)
- processor_dynamic.py (experimental version)
- processor_enhanced.py (functionality merged into processor.py)
- bmi_validator.py (merged into processor.py)
- threshold_calculator.py (merged into processor.py)
- dynamic_reset_manager.py (merged into processor.py)

## Implementation Plan (No Code)

### Phase 1: Analysis & Backup
1. Create backup of entire src/ directory
2. Analyze all test imports to create migration map
3. Document all public APIs that need to be preserved
4. Identify exact functions/classes used by main.py and tests

### Phase 2: Consolidation
1. **Merge validation classes into processor.py**:
   - Copy BMIValidator class and methods
   - Copy ThresholdCalculator class and methods
   - Copy DynamicResetManager class and methods
   - Ensure no circular dependencies

2. **Rename processor_database.py to database.py**:
   - Simple file rename
   - Update all imports in processor.py

3. **Update import statements**:
   - Update main.py imports
   - Create import compatibility mapping for tests
   - Use find/replace for bulk import updates

### Phase 3: Testing & Verification
1. Run main.py with test data to verify functionality
2. Run all tests and fix import errors
3. Verify visualization generation still works
4. Test batch reprocessing functionality

### Phase 4: Cleanup
1. Remove old/unused files
2. Update any documentation references
3. Verify no broken imports remain
4. Clean up any temporary compatibility code

## Validation & Rollout

### Test Strategy
- Unit tests: Verify all moved classes/functions work correctly
- Integration tests: Ensure main.py processes data correctly
- Import tests: Check all test files can import needed functions
- Regression tests: Compare output before/after consolidation

### Manual QA Checklist
- [ ] main.py runs without import errors
- [ ] All visualization features work
- [ ] Batch reprocessing works correctly
- [ ] All existing tests pass
- [ ] No circular import issues
- [ ] File sizes remain manageable (<1000 lines each)

### Rollout Plan
1. Create feature branch for cleanup
2. Implement changes incrementally with commits for each phase
3. Run full test suite after each phase
4. Manual testing with sample data
5. Merge to main after verification

## Risks & Mitigations

### Risk 1: Breaking Test Dependencies
- **Impact**: High - Many tests depend on specific imports
- **Mitigation**: Create detailed import migration map before starting
- **Recovery**: Keep backup of original structure

### Risk 2: Circular Import Issues
- **Impact**: Medium - Merging files might create circular dependencies
- **Mitigation**: Careful analysis of import chains, use lazy imports if needed
- **Recovery**: Refactor to break circular dependencies

### Risk 3: Large File Sizes
- **Impact**: Low - Merged processor.py might become too large
- **Mitigation**: Monitor file size, consider keeping more utility files if needed
- **Recovery**: Split back into feature-specific modules

### Risk 4: Lost Git History
- **Impact**: Low - File deletions lose git blame history
- **Mitigation**: Document consolidation in commit messages
- **Recovery**: Git history still available in previous commits

## Acceptance Criteria
- [ ] Src directory contains maximum 4 files
- [ ] All current functionality preserved
- [ ] No duplicate code across files
- [ ] All tests pass without modification (except imports)
- [ ] main.py runs successfully with real data
- [ ] Clear separation of concerns maintained
- [ ] No performance degradation

## Out of Scope
- Refactoring algorithm logic
- Changing public APIs
- Modifying test logic (only import updates)
- Updating configuration structure
- Performance optimizations
- Adding new features

## Open Questions
1. Should we keep reprocessor.py or merge it into processor.py?
   - Recommendation: Keep it separate for clarity
2. Should database.py keep the "processor" prefix?
   - Recommendation: No, simpler is better
3. Should we create an __init__.py for cleaner imports?
   - Recommendation: Not necessary with current structure
4. Should we add type hints during consolidation?
   - Recommendation: No, keep changes minimal

## Review Cycle
### Self-Review Notes
- Verified all imports in main.py and major test files
- Confirmed consolidation strategy maintains separation of concerns
- File size estimates: processor.py (~1200 lines), database.py (~400 lines), visualization.py (~600 lines)
- Risk assessment complete with mitigation strategies

### Revisions
- Added reprocessor.py to final structure (initially missed its importance for batch operations)
- Clarified that processor.py will remain stateless despite consolidation
- Added import compatibility mapping step for smoother test migration