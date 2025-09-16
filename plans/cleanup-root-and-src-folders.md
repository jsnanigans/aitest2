# Plan: Clean Up Root and Src Folders

## Summary
Organize and consolidate the root directory and src folder to improve project structure, reduce clutter, and enhance maintainability by removing obsolete files and consolidating related functionality.

## Context
- Source: User request for cleanup
- Assumptions:
  - Scripts folder contains many one-off test/debug scripts that may be obsolete
  - Some visualization modules (viz_enhanced, viz_index, viz_reset_insights) may be redundant
  - Reset manager and adaptive Kalman modules may need consolidation
  - Documentation is scattered and needs organization
- Constraints:
  - Must maintain backward compatibility for main.py entry point
  - Core architecture (stateless processor) must be preserved
  - All active functionality must remain accessible

## Requirements
### Functional
- Preserve all active functionality
- Maintain existing API surface for external consumers
- Keep main.py as primary entry point
- Preserve test suite integrity

### Non-functional
- Improve code organization and discoverability
- Reduce file count and complexity
- Enhance maintainability
- Clear separation of concerns

## Alternatives

### Option A: Minimal Cleanup
- Approach: Remove only clearly obsolete files, keep structure mostly intact
- Pros:
  - Low risk of breaking changes
  - Quick to implement
  - Easy to review
- Cons:
  - Limited improvement
  - Doesn't address structural issues
  - Scripts folder remains cluttered
- Risks: Minimal

### Option B: Moderate Reorganization
- Approach: Consolidate related modules, archive old scripts, organize docs
- Pros:
  - Significant improvement in organization
  - Preserves useful utilities
  - Balanced risk/reward
- Cons:
  - Some import path changes needed
  - Requires careful testing
- Risks: Medium - potential for missed dependencies

### Option C: Full Restructure
- Approach: Complete reorganization with new folder structure
- Pros:
  - Optimal organization
  - Clear separation of concerns
  - Future-proof structure
- Cons:
  - High implementation effort
  - Many breaking changes
  - Extensive testing required
- Risks: High - significant potential for breakage

## Recommendation
**Option B: Moderate Reorganization** - provides the best balance of improvement vs risk. Focus on consolidating visualization modules, archiving obsolete scripts, and organizing documentation.

## High-Level Design

### Root Directory Changes
```
strem_process_anchor/
├── src/                    # Core source (consolidated)
├── tests/                  # Test suite (unchanged)
├── data/                   # Sample data (unchanged)
├── docs/                   # All documentation (reorganized)
├── scripts/                # Active utility scripts only
│   └── archive/           # Old/obsolete scripts
├── plans/                  # Planning documents (unchanged)
├── .knowledge/            # Knowledge base (unchanged)
├── config.toml            # Configuration
├── requirements.txt       # Dependencies
├── main.py               # Entry point
├── README.md             # Project overview
├── pyrightconfig.json    # Type checking config
└── .gitignore           # Git ignore rules
```

### Src Directory Consolidation
```
src/
├── __init__.py           # Package exports
├── processor.py          # Core processor (unchanged)
├── database.py           # State persistence (unchanged)
├── kalman.py            # Unified Kalman implementation
├── validation.py         # All validation logic
├── quality_scorer.py     # Quality scoring
├── visualization.py      # Unified visualization
├── constants.py          # All constants
└── utils.py             # Utility functions
```

### Files to Remove/Archive
- `src/kalman_adaptive.py` → merge into `src/kalman.py`
- `src/reset_manager.py` → merge into `src/processor.py` or `src/kalman.py`
- `src/viz_enhanced.py` → merge into `src/visualization.py`
- `src/viz_index.py` → merge into `src/visualization.py`
- `src/viz_reset_insights.py` → merge into `src/visualization.py`
- `api.md` → move to `docs/api.md`
- `AGENTS.md` → move to `docs/AGENTS.md`

## Implementation Plan (No Code)

### Phase 1: Scripts Cleanup
1. Identify active vs obsolete scripts
   - Review each script in `scripts/` for current relevance
   - Check for dependencies from main code
   - Identify test scripts vs utility scripts
2. Create `scripts/archive/` directory
3. Move obsolete scripts to archive
4. Keep only essential utilities in `scripts/`:
   - `preprocess_csv.py` (data preprocessing)
   - `generate_index.py` (index generation)
   - `batch_visualize.py` (batch processing)
   - `measure_performance.py` (performance testing)

### Phase 2: Src Module Consolidation
1. Merge visualization modules
   - Analyze dependencies on viz_* modules
   - Consolidate all visualization code into `visualization.py`
   - Update imports in processor.py and main.py
   - Ensure all visualization features remain accessible
2. Merge Kalman-related modules
   - Combine `kalman_adaptive.py` into `kalman.py`
   - Move reset management logic to appropriate location
   - Update all Kalman-related imports
3. Update `src/__init__.py` exports
   - Remove references to deleted modules
   - Ensure all public APIs remain exported

### Phase 3: Documentation Organization
1. Move root-level docs to `docs/`
   - `api.md` → `docs/api.md`
   - `AGENTS.md` → `docs/AGENTS.md`
2. Create documentation index
   - `docs/README.md` with navigation
   - Group docs by category (API, Development, Architecture)
3. Update root README.md
   - Simplify to essential information
   - Link to detailed docs in `docs/`

### Phase 4: Testing and Validation
1. Run full test suite after each phase
2. Test main.py with sample data
3. Verify all imports resolve correctly
4. Check visualization outputs remain unchanged
5. Validate batch processing scripts still work

## Validation & Rollout

### Test Strategy
- Unit tests: Ensure all test files pass
- Integration tests: Process sample CSV files
- Visualization tests: Generate plots, verify output
- Import tests: Check all public APIs accessible
- Performance tests: Ensure no regression

### Manual QA Checklist
- [ ] main.py processes test data correctly
- [ ] All visualization modes work
- [ ] Database persistence functions
- [ ] Quality scoring produces same results
- [ ] Kalman filtering unchanged
- [ ] Scripts in scripts/ folder execute
- [ ] Documentation links work
- [ ] No broken imports

### Rollout Plan
1. Create feature branch for cleanup
2. Implement Phase 1 (scripts), test, commit
3. Implement Phase 2 (src consolidation), test, commit
4. Implement Phase 3 (docs), test, commit
5. Full regression testing
6. Create PR with detailed change summary
7. Merge after review

## Risks & Mitigations

### Risk 1: Breaking Hidden Dependencies
- **Mitigation**: Comprehensive grep search before removing any file
- **Mitigation**: Keep archive folder for 30 days before permanent deletion

### Risk 2: Import Path Changes Break External Code
- **Mitigation**: Maintain compatibility imports in __init__.py
- **Mitigation**: Document all import changes clearly

### Risk 3: Lost Functionality from Consolidation
- **Mitigation**: Create detailed mapping of functions before/after
- **Mitigation**: Extensive testing of all features

## Acceptance Criteria
- [ ] Root directory has clear, logical structure
- [ ] Src folder contains only essential modules (≤10 files)
- [ ] Scripts folder contains only active utilities (≤10 files)
- [ ] All tests pass without modification
- [ ] main.py works with all existing command-line options
- [ ] No functionality is lost
- [ ] Documentation is organized and accessible
- [ ] File count reduced by at least 40%

## Out of Scope
- Refactoring core processor logic
- Changing public API signatures
- Modifying test structure
- Updating dependencies
- Performance optimizations
- Adding new features

## Open Questions
1. Should we keep all test_* scripts in scripts/ or move to tests/?
2. Are there any external systems depending on specific file paths?
3. Should archived scripts be kept in repo or removed entirely?
4. Do we need to maintain any deprecated imports for compatibility?

## Review Cycle
### Self-Review Notes
- Verified all active functionality identified
- Consolidation plan preserves all features
- Risk mitigation strategies are comprehensive
- Testing plan covers all critical paths

### Revisions
- Added explicit archive strategy for scripts
- Clarified which scripts to keep as active utilities
- Added compatibility import consideration
- Specified 30-day archive retention before deletion