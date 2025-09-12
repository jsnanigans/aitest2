# Plan: Clean Up and Refactor src Directory

## Summary
Comprehensive refactoring of the weight stream processor source code to improve maintainability, readability, and code organization. The codebase has grown organically with many features added incrementally, resulting in large monolithic files (processor.py has 2,075 lines) with mixed responsibilities and duplicated logic. Goal is to create a clean, modular architecture while preserving all functionality.

## Context
- Source: User request for code cleanup and refactoring
- Current state: 4 main files in src/ totaling ~3,700 lines, plus 6 backup/experimental files
- Main issues identified:
  - processor.py is 2,075 lines with 6+ classes and mixed responsibilities
  - Multiple validation and quality components mixed together
  - Duplicated threshold calculation logic
  - Complex nested conditionals throughout
  - No clear separation of concerns
- Assumptions:
  - Functionality must be preserved (no breaking changes)
  - Tests must continue to pass
  - Performance should not degrade
  - Architecture constraints (stateless processor) must be maintained

## Requirements
### Functional
- Preserve all existing functionality
- Maintain backward compatibility with existing tests
- Keep stateless architecture for WeightProcessor
- Maintain separation between processing logic and state management
- Preserve all validation and quality features (BMI, physiological limits, etc.)

### Non-functional
- Reduce file sizes (target: <500 lines per file where possible)
- Eliminate code duplication
- Improve separation of concerns
- Standardize naming conventions
- Improve code readability and maintainability
- Clear module boundaries

## Alternatives

### Option A: Minimal Consolidation (Quick Fix)
- Approach: Simply remove backup files and consolidate into 3-4 main files
- Pros:
  - Quick to implement (2-3 hours)
  - Minimal risk of breaking changes
  - Easy to review and test
- Cons:
  - Doesn't address fundamental issues
  - processor.py remains at 2,075 lines
  - Code duplication persists
  - Mixed responsibilities continue
- Risks: Technical debt continues to grow

### Option B: Moderate Refactoring (Balanced)
- Approach: Create logical modules within src/ (6-8 files) with clear responsibilities
- Pros:
  - Good balance of effort and improvement
  - Manageable file sizes (300-500 lines)
  - Clear separation of concerns
  - Easier to test and maintain
- Cons:
  - More files than current structure
  - Requires updating all imports
  - 8-10 hours of work
- Risks: Import compatibility issues during migration

### Option C: Full Modular Architecture (Comprehensive)
- Approach: Create proper package structure with submodules (15-20 files)
- Pros:
  - Best long-term maintainability
  - Excellent separation of concerns
  - Each file has single responsibility
  - Highly testable
- Cons:
  - Significant effort (15-20 hours)
  - Many import changes needed
  - More complex directory structure
- Risks: Over-engineering for current needs

## Recommendation
**Option B: Moderate Refactoring (Balanced)**

Rationale:
- Addresses the core problems without over-engineering
- Reduces processor.py from 2,075 lines to ~300 lines
- Creates maintainable file sizes (300-500 lines each)
- Clear separation makes future changes easier
- Reasonable effort (8-10 hours) with good ROI
- Lower risk than full restructuring while still providing significant improvements

## High-Level Design

### Target Structure
```
src/
├── processor.py         # Core weight processing orchestration (~300 lines)
├── kalman.py           # Kalman filter logic (~250 lines)
├── validation.py       # All validation logic (BMI, physiological, thresholds) (~400 lines)
├── quality.py          # Data quality and preprocessing (~400 lines)
├── database.py         # State persistence (~270 lines)
├── reprocessor.py      # Batch reprocessing (~440 lines)
├── visualization.py    # Dashboard and charts (~900 lines)
└── models.py          # Data models and constants (~200 lines)
```

### Module Responsibilities

1. **processor.py** (Orchestration Layer):
   - WeightProcessor class with main process_weight method
   - Orchestrates validation, quality, and Kalman filtering
   - Delegates to specialized modules
   - Maintains stateless architecture

2. **kalman.py** (Filter Logic):
   - Extract from processor: _initialize_kalman_immediate, _update_kalman_state
   - Kalman filter configuration and operations
   - State prediction and update logic
   - Innovation and confidence calculations

3. **validation.py** (Validation Layer):
   - BMIValidator class (from current processor.py)
   - ThresholdCalculator class (from current processor.py)
   - Physiological limits validation
   - Weight validation methods
   - Rejection categorization

4. **quality.py** (Data Quality Layer):
   - DataQualityPreprocessor class
   - AdaptiveOutlierDetector class
   - SourceQualityMonitor class
   - AdaptiveKalmanConfig class
   - Unit conversion and BMI detection

5. **database.py** (Persistence Layer):
   - Current ProcessorStateDB (no changes needed)
   - State serialization/deserialization
   - Snapshot management

6. **models.py** (Data Structures):
   - ThresholdResult class
   - DynamicResetManager class
   - Constants (SOURCE_PROFILES, etc.)
   - Helper functions (categorize_rejection, etc.)

### Dependency Flow
```
main.py
   ↓
processor.py → kalman.py
   ↓        ↘
validation.py  quality.py
   ↓              ↓
models.py ← database.py
```

## Implementation Plan (No Code)

### Phase 1: Preparation (1 hour)
1. Create backup of entire src/ directory to src_backup_[timestamp]/
2. Create detailed import dependency map from tests and main.py
3. Document all public APIs that must be preserved
4. Set up test harness to verify functionality after each step

### Phase 2: Extract Models and Constants (1 hour)
1. Create models.py file
2. Move ThresholdResult class from processor.py
3. Move all constants (SOURCE_PROFILES, etc.)
4. Move helper functions (categorize_rejection, get_rejection_severity)
5. Update imports in processor.py
6. Run tests to verify

### Phase 3: Extract Kalman Logic (1.5 hours)
1. Create kalman.py file
2. Extract Kalman-specific methods from WeightProcessor:
   - _initialize_kalman_immediate
   - _update_kalman_state
   - _calculate_confidence
   - _create_result (Kalman-specific parts)
3. Create KalmanFilterManager class to encapsulate logic
4. Update WeightProcessor to use KalmanFilterManager
5. Run tests to verify

### Phase 4: Extract Validation Logic (2 hours)
1. Create validation.py file
2. Move BMIValidator class from processor.py
3. Move ThresholdCalculator class from processor.py
4. Move validation methods from WeightProcessor:
   - _validate_weight
   - _get_physiological_limit
5. Consolidate duplicate validation logic
6. Update processor.py imports
7. Run tests to verify

### Phase 5: Extract Quality Components (2 hours)
1. Create quality.py file
2. Move DataQualityPreprocessor class
3. Move AdaptiveOutlierDetector class
4. Move SourceQualityMonitor class
5. Move AdaptiveKalmanConfig class
6. Move process_weight_enhanced wrapper logic
7. Update processor.py to use quality module
8. Run tests to verify

### Phase 6: Refactor Core Processor (1.5 hours)
1. Simplify WeightProcessor class:
   - Keep only process_weight orchestration
   - Delegate to validation, quality, kalman modules
   - Remove all moved methods
2. Clean up imports and organization
3. Add clear module docstrings
4. Ensure stateless architecture maintained
5. Run tests to verify

### Phase 7: Update Imports and Tests (1 hour)
1. Update main.py imports
2. Update all test file imports systematically
3. Add backward compatibility imports if needed
4. Run full test suite
5. Fix any remaining import issues

### Phase 8: Cleanup and Documentation (1 hour)
1. Remove backup/experimental files:
   - src_backup_* directories
   - Any other legacy files
2. Update code comments and docstrings
3. Verify all functionality with manual testing
4. Create migration notes for future reference

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
- [ ] processor.py reduced from 2,075 to ~300 lines
- [ ] No file exceeds 500 lines (except visualization.py at ~900)
- [ ] All current functionality preserved
- [ ] No duplicate code across files
- [ ] All tests pass (with import updates only)
- [ ] main.py runs successfully with real data
- [ ] Clear separation of concerns achieved
- [ ] No performance degradation
- [ ] Each module has single, clear responsibility
- [ ] Dependency flow is unidirectional (no circular imports)

## Out of Scope
- Refactoring algorithm logic
- Changing public APIs
- Modifying test logic (only import updates)
- Updating configuration structure
- Performance optimizations
- Adding new features

## Open Questions
1. Should we keep visualization.py at 900 lines or split it further?
   - Recommendation: Keep as-is for now, it's cohesive
2. Should DynamicResetManager stay in models.py or move to validation.py?
   - Recommendation: models.py since it's more of a manager than validator
3. Should we create an __init__.py for cleaner imports?
   - Recommendation: Yes, to maintain backward compatibility
4. Should we add type hints during refactoring?
   - Recommendation: Only where it improves clarity, not comprehensive
5. Is 8 modules too many compared to current 4?
   - Recommendation: The clarity gained outweighs the file count increase

## Code Quality Improvements

### Specific Issues to Address

1. **Deep Nesting in processor.py**:
   - _process_weight_internal has 6+ levels of nesting
   - Extract guard clauses and early returns
   - Break into smaller, focused methods

2. **Duplicate Threshold Logic**:
   - Threshold calculations appear in 3+ places
   - Consolidate into single source of truth
   - Use consistent units (percentage vs kg)

3. **Mixed Responsibilities**:
   - WeightProcessor handles processing, validation, and quality
   - Separate concerns into appropriate modules
   - Clear interfaces between modules

4. **Complex Conditionals**:
   - Many if/elif chains with complex conditions
   - Extract to named boolean methods
   - Use strategy pattern where appropriate

5. **Inconsistent Naming**:
   - Mix of camelCase and snake_case
   - Abbreviations (ni, pct, etc.)
   - Standardize throughout

## Timeline Estimate

- Phase 1 (Preparation): 1 hour
- Phase 2 (Models): 1 hour  
- Phase 3 (Kalman): 1.5 hours
- Phase 4 (Validation): 2 hours
- Phase 5 (Quality): 2 hours
- Phase 6 (Processor): 1.5 hours
- Phase 7 (Imports): 1 hour
- Phase 8 (Cleanup): 1 hour

**Total: 11 hours**

## Review Cycle
### Self-Review Notes
- Analyzed processor.py line-by-line to identify extraction points
- Verified module boundaries avoid circular dependencies
- Confirmed all test imports can be updated systematically
- File size targets are achievable with proposed structure
- Risk mitigation strategies are comprehensive

### Revisions After Analysis
- Added kalman.py as separate module (initially part of processor)
- Moved DynamicResetManager to models.py (better fit than validation)
- Increased time estimate from 8-10 to 11 hours based on code analysis
- Added specific code quality improvements section
- Clarified dependency flow diagram