# Plan: Clean Up src Directory - Visualization Consolidation Focus

## Summary
Consolidate and reorganize the src directory to eliminate redundancy, improve maintainability, and establish clear module boundaries. The current structure has significant duplication in visualization modules (12 viz_* files with ~5,845 lines) and needs streamlining.

## Context
- Source: User request to clean up src directory after reviewing all files
- Current state: 21 files in src/ directory
  - Core processing: 6 files (~2,500 lines)
  - Visualization: 12 viz_* files (~5,845 lines) + main visualization.py (954 lines)
  - Utilities: 3 files (logging, constants, init)
- Main issues identified:
  - Massive duplication in visualization (12 separate viz files!)
  - Multiple dashboard implementations (diagnostic has 3 versions)
  - Temporary patch file (viz_diagnostic_fix.py)
  - Unclear which visualization is primary
  - Router pattern adds unnecessary complexity
- Assumptions:
  - No backward compatibility required (per guidelines)
  - Prefer updating existing tools over creating new ones
  - Maintain stateless processor architecture
  - Use latest Python 3.11+ syntax

## Requirements
### Functional
- Maintain all current functionality
- Preserve stateless processing architecture
- Keep core processing pipeline intact
- Ensure tests continue to pass
- Consolidate visualization features into unified interface

### Non-functional
- Reduce visualization files from 12 to 1-2 maximum
- Eliminate code duplication (target: 50%+ reduction)
- Improve module organization and naming
- Establish clear separation of concerns
- Simplify import structure
- Remove temporary/patch files

## Alternatives

### Option A: Minimal Consolidation
- Approach: Just delete unused viz files, keep the rest
- Pros:
  - Very quick (1 hour)
  - Low risk
  - Minimal test changes
- Cons:
  - Doesn't fix duplication
  - Still have 5-6 viz files
  - Confusion remains
- Risks: Problem persists

### Option B: Full Restructure with Subpackages
- Approach: Create subpackages (core/, viz/, utils/)
- Pros:
  - Clear separation of concerns
  - Scalable structure
  - Better organization
- Cons:
  - More complex refactoring
  - All imports need updating
  - Over-engineering for project size
- Risks: Higher chance of breaking changes

### Option C: Targeted Visualization Consolidation
- Approach: Merge all viz files into 1-2 files, clean up core modules
- Pros:
  - Addresses main pain point (12 viz files)
  - Significant code reduction
  - Maintains flat structure
  - Reasonable effort (4-6 hours)
- Cons:
  - visualization.py will be larger (~1500 lines)
  - Some complexity in merged file
- Risks: Need careful merging to avoid losing features

## Recommendation
**Option C: Targeted Visualization Consolidation**

Rationale:
- Directly addresses the biggest problem (12 visualization files)
- Reduces total files from 21 to ~10
- Eliminates ~3,000 lines of duplicate code
- Maintains simplicity with flat structure
- Reasonable effort (4-6 hours) with high impact
- Low risk since core processing untouched

## High-Level Design

### Target Structure
```
src/
├── __init__.py              # Public API exports
├── constants.py             # All constants and profiles (keep as-is)
├── database.py              # State persistence (keep as-is)
├── kalman.py               # Kalman filter logic (keep as-is)
├── processor.py            # Core processing (keep as-is)
├── quality_scorer.py       # Quality scoring (keep as-is)
├── validation.py           # All validation logic (keep as-is)
├── visualization.py        # Unified visualization (~1500 lines)
└── utils.py                # Logging and utilities (merge from logging_utils)
```

### Files to Remove
```
src/
├── viz_constants.py        → merge into visualization.py
├── viz_diagnostic.py       → merge into visualization.py
├── viz_diagnostic_fix.py   → DELETE (temporary patch)
├── viz_diagnostic_simple.py → merge into visualization.py
├── viz_export.py          → merge into visualization.py
├── viz_index.py           → merge into visualization.py
├── viz_kalman.py          → merge into visualization.py
├── viz_logger.py          → merge into utils.py
├── viz_plotly.py          → merge into visualization.py
├── viz_plotly_enhanced.py → merge into visualization.py
├── viz_quality.py         → merge into visualization.py
├── viz_router.py          → merge into visualization.py
└── logging_utils.py       → merge into utils.py
```

### Consolidated Module Design

#### visualization.py Structure
```python
# Single unified visualization module with clear sections:

# 1. Constants and Configuration
CHART_COLORS = {...}
PLOTLY_CONFIG = {...}

# 2. Base Dashboard Class
class BaseDashboard:
    """Common functionality for all dashboards"""

# 3. Dashboard Implementations  
class DiagnosticDashboard(BaseDashboard):
    """Unified diagnostic dashboard (merge 3 versions)"""

class InteractiveDashboard(BaseDashboard):
    """Plotly interactive dashboard (merge 2 versions)"""

class StaticDashboard(BaseDashboard):
    """Matplotlib static dashboard"""

# 4. Specialized Visualizations
class KalmanVisualizer:
    """Kalman-specific charts"""

class QualityVisualizer:
    """Quality score visualizations"""

class IndexVisualizer:
    """Index/overview visualizations"""

# 5. Export Functions
def export_dashboard(...)
def export_to_pdf(...)
def export_to_html(...)

# 6. Router/Factory
def create_dashboard(results, user_id, config):
    """Smart router that picks appropriate dashboard"""
```

#### utils.py Structure
```python
# Merged utilities module

# 1. Logging
class StructuredLogger:
    """From logging_utils.py"""

# 2. Performance
class PerformanceTimer:
    """From logging_utils.py"""

# 3. Visualization Utilities
def get_logger():
    """From viz_logger.py"""

# 4. General Utilities
def format_timestamp(...)
def safe_divide(...)
```

## Implementation Plan (No Code)

### Phase 1: Analysis and Backup (30 min)
1. Create backup of src/ directory
2. Map all imports from viz_* files
3. Identify unique vs duplicate functionality
4. Document which tests use which viz modules
5. Create function mapping spreadsheet

### Phase 2: Create Unified Visualization Structure (1 hour)
1. Start with current visualization.py as base
2. Create class hierarchy:
   - BaseDashboard abstract class
   - DiagnosticDashboard (merge 3 versions)
   - InteractiveDashboard (merge plotly versions)
   - StaticDashboard (existing matplotlib)
3. Add specialized visualizer classes:
   - KalmanVisualizer
   - QualityVisualizer
   - IndexVisualizer
4. Set up configuration constants section

### Phase 3: Merge Diagnostic Dashboards (1 hour)
1. Compare viz_diagnostic.py, viz_diagnostic_simple.py
2. Identify common functionality
3. Create single configurable DiagnosticDashboard class
4. Merge unique features from each version
5. Remove viz_diagnostic_fix.py patch logic
6. Test diagnostic dashboard generation

### Phase 4: Merge Plotly Implementations (45 min)
1. Compare viz_plotly.py and viz_plotly_enhanced.py
2. Merge into single InteractiveDashboard class
3. Add configuration flags for enhanced features
4. Consolidate plotly configuration
5. Test interactive dashboard generation

### Phase 5: Integrate Specialized Visualizations (45 min)
1. Extract KalmanVisualizer from viz_kalman.py
2. Extract QualityVisualizer from viz_quality.py
3. Extract IndexVisualizer from viz_index.py
4. Integrate as methods/classes in visualization.py
5. Ensure they work with main dashboards

### Phase 6: Merge Router and Export Logic (30 min)
1. Extract router logic from viz_router.py
2. Implement as create_dashboard factory function
3. Merge export functions from viz_export.py
4. Add smart mode detection (interactive vs static)
5. Test routing logic with different configs

### Phase 7: Consolidate Utilities (30 min)
1. Create utils.py from logging_utils.py
2. Add viz_logger.py content
3. Add any utility functions from viz_constants.py
4. Update all imports to use utils.py
5. Test logging functionality

### Phase 8: Update Imports and Clean Up (1 hour)
1. Update __init__.py exports
2. Fix imports in processor.py, main.py
3. Update all test file imports
4. Delete all viz_*.py files
5. Delete logging_utils.py
6. Run full test suite

### Phase 9: Testing and Validation (30 min)
1. Run all unit tests
2. Test each visualization mode
3. Verify data processing unchanged
4. Test with sample CSV files
5. Check for any missing functionality

## Validation & Rollout

### Test Strategy
- Run existing test suite after each phase
- Test all visualization modes:
  - Static (matplotlib)
  - Interactive (plotly)
  - Diagnostic (unified version)
- Verify data processing unchanged
- Test with all sample CSV files
- Check HTML output generation

### Manual QA Checklist
- [ ] main.py processes CSV files correctly
- [ ] Static visualization works
- [ ] Interactive dashboard generates
- [ ] Diagnostic dashboard displays properly
- [ ] Export functions work (HTML, PDF if configured)
- [ ] Kalman visualizations render
- [ ] Quality score charts display
- [ ] Index viewer works (if used)
- [ ] All tests pass
- [ ] No import errors

### Rollout Plan
1. Create feature branch: `cleanup-viz-consolidation`
2. Implement phases with commits after each
3. Run tests continuously during development
4. Manual testing with test data files
5. Create before/after comparison
6. Merge to main branch

## Risks & Mitigations

### Risk 1: Lost Visualization Features
- **Impact**: High - Some unique features might be missed during merge
- **Mitigation**: Careful feature mapping before consolidation
- **Recovery**: Backup allows restoration of specific features

### Risk 2: Large visualization.py File
- **Impact**: Medium - File will be ~1500 lines
- **Mitigation**: Good class organization and clear sections
- **Recovery**: Can split into viz_core.py and viz_components.py if needed

### Risk 3: Test Import Failures
- **Impact**: Medium - Tests import from specific viz modules
- **Mitigation**: Update imports systematically, test continuously
- **Recovery**: Add compatibility imports in __init__.py

### Risk 4: Configuration Compatibility
- **Impact**: Low - Config might reference old module names
- **Mitigation**: Add fallback handling in router logic
- **Recovery**: Update config.toml files

## Acceptance Criteria
- [ ] Reduced from 21 to ~9 files in src/
- [ ] Eliminated all 12 viz_*.py files
- [ ] Single visualization.py handles all viz needs
- [ ] Removed ~3,000 lines of duplicate code
- [ ] All tests pass
- [ ] All visualization modes work
- [ ] No functionality lost
- [ ] Clear module boundaries
- [ ] Simplified import structure
- [ ] No temporary/patch files remain

## Out of Scope
- Refactoring core processing logic
- Changing algorithms
- Database schema changes
- Performance optimizations
- Adding new features
- Modifying test logic (only import updates)

## Open Questions
1. Should visualization.py be split if it exceeds 1500 lines?
   - Recommendation: Keep unified for now, split only if it exceeds 2000 lines
2. Should we keep viz_export.py separate for modularity?
   - Recommendation: No, merge it for consistency
3. Is the index viewer functionality actively used?
   - Recommendation: Keep it but as optional feature
4. Should quality_scorer.py be merged with validation.py?
   - Recommendation: Keep separate, they have different purposes

## Timeline Estimate

- Phase 1 (Analysis): 30 minutes
- Phase 2 (Structure): 1 hour
- Phase 3 (Diagnostic): 1 hour
- Phase 4 (Plotly): 45 minutes
- Phase 5 (Specialized): 45 minutes
- Phase 6 (Router/Export): 30 minutes
- Phase 7 (Utilities): 30 minutes
- Phase 8 (Imports): 1 hour
- Phase 9 (Testing): 30 minutes

**Total: 6 hours**

## Review Cycle
### Self-Review Notes
- Reviewed all 21 files in src/ directory
- Identified 12 visualization files as primary cleanup target
- Visualization has most duplication (~5,845 lines across 12 files)
- Core processing modules are relatively well-organized
- Focus on consolidation over restructuring

### Revisions After Analysis
- Changed focus from processor.py refactoring to visualization consolidation
- Reduced scope to target highest-impact improvements
- Decreased time estimate from 11 to 6 hours
- Prioritized removing duplicate dashboard implementations
- Added specific file removal list