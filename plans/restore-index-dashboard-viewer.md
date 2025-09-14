# Plan: Restore Index Dashboard Viewer with User List and IFrame

## Summary
Restore the missing index.html functionality that provides a user list sidebar with sorting capabilities and an iframe-based dashboard viewer. The current implementation creates a plotly-based index_dashboard.html instead of the expected index.html with user navigation.

## Context
- **Source**: User request to restore previous index.html functionality
- **Current Issue**: The index viewer is missing/broken - it should show a list of users on the side with stats, allow sorting, and display individual reports in an iframe
- **Previous Implementation**: Found in `src_backup_20250914_161137/viz_index.py` which has the complete implementation
- **Assumptions**:
  - The backup implementation is the desired functionality
  - Need to integrate with current visualization.py module
  - Should work with existing dashboard generation

## Requirements

### Functional Requirements
1. Generate index.html with user list sidebar
2. Display user statistics (total values, accepted, rejected)
3. Sort users by multiple criteria (ID, total, accepted, rejected, acceptance rate)
4. Click user to load their dashboard in iframe
5. Keyboard navigation with up/down arrow keys
6. Search/filter users by ID
7. Quick sort buttons for common sorting needs

### Non-functional Requirements
- Standalone HTML file (no server required)
- Fast sorting and filtering
- Smooth iframe loading
- Responsive design
- Works offline with file:// protocol

## Alternatives

### Option A: Restore Backup Implementation
**Approach**: Copy viz_index.py from backup and integrate into visualization.py
- **Pros**: 
  - Complete working implementation exists
  - All features already implemented
  - Tested and verified to work
- **Cons**: 
  - May have redundancy with current code
  - Need to ensure compatibility with current structure
- **Risks**: Minimal - backup code is proven to work

### Option B: Rewrite from Scratch
**Approach**: Create new implementation in visualization.py
- **Pros**: 
  - Clean integration with current code
  - Can optimize for current architecture
- **Cons**: 
  - Time consuming
  - Risk of missing features
  - Need to recreate all functionality
- **Risks**: High - may introduce bugs or miss features

### Option C: Hybrid Approach
**Approach**: Extract key functions from backup and integrate into IndexVisualizer class
- **Pros**: 
  - Leverages existing code
  - Maintains current architecture
  - Can optimize during integration
- **Cons**: 
  - More complex integration
  - Need to map between implementations
- **Risks**: Medium - need careful integration

## Recommendation
**Option A: Restore Backup Implementation** - The backup contains a complete, working implementation with all requested features. It's the fastest and most reliable approach.

## High-Level Design

### Architecture Overview
```
visualization.py
├── IndexVisualizer class (modified)
│   ├── create_index_from_results() - Entry point
│   ├── extract_user_stats() - Get stats from results
│   ├── find_dashboard_files() - Locate HTML files
│   ├── generate_summary_stats() - Calculate totals
│   ├── generate_css() - Embedded styles
│   ├── generate_javascript() - Dashboard viewer logic
│   └── generate_index_html() - Build complete HTML
│
main.py
└── Calls create_index_from_results() after visualization
```

### Data Flow
1. Results JSON → extract_user_stats() → user statistics
2. User stats + viz directory → find_dashboard_files() → locate dashboards
3. User stats → generate_summary_stats() → overall statistics
4. All data → generate_index_html() → index.html with embedded data
5. Browser loads index.html → JavaScript creates interactive viewer

### Key Components
- **User List**: Sidebar with sortable, searchable user list
- **Dashboard Viewer**: IFrame container for loading dashboards
- **Sort Controls**: Dropdown + direction toggle + quick sort buttons
- **Keyboard Handler**: Arrow key navigation between users
- **Search Filter**: Real-time user filtering

## Implementation Plan (No Code)

### Step 1: Update IndexVisualizer Class
- Replace current IndexVisualizer implementation in visualization.py
- Import required functions from backup viz_index.py
- Maintain backward compatibility with existing calls

### Step 2: Add Helper Methods
- extract_user_stats(): Parse results JSON for user statistics
- find_dashboard_files(): Locate dashboard HTML files for each user
- generate_summary_stats(): Calculate overall statistics
- generate_css(): Return embedded CSS styles
- generate_javascript(): Return DashboardViewer JavaScript class
- generate_index_html(): Build complete HTML document

### Step 3: Update create_index_from_results()
- Change from plotly dashboard to HTML index generation
- Process all users from results
- Find dashboard files in output directory
- Generate index.html with embedded data

### Step 4: Ensure Main.py Integration
- Verify main.py calls create_index_from_results correctly
- Pass proper parameters (results dict, output directory)
- Display index.html path in output

### Step 5: Test File Discovery
- Ensure dashboard file patterns are correct:
  - {user_id}.html
  - dashboard_enhanced_{user_id}.html
  - dashboard_{user_id}.html
  - viz_{user_id}.html

## Validation & Rollout

### Test Strategy
1. **Unit Tests**:
   - Test extract_user_stats with various result formats
   - Test find_dashboard_files with different naming patterns
   - Test summary statistics calculation

2. **Integration Tests**:
   - Generate index from sample results
   - Verify HTML structure
   - Check JavaScript functionality
   - Test with missing dashboards

3. **Manual Testing**:
   - Load index.html in browser
   - Test all sorting options
   - Verify search filtering
   - Test keyboard navigation
   - Check iframe loading
   - Test with 5, 50, 100 users

### Manual QA Checklist
- [ ] Index.html generates successfully
- [ ] User list displays all users
- [ ] Statistics show correctly (total/accepted/rejected)
- [ ] Sort dropdown works for all options
- [ ] Sort direction toggle works
- [ ] Quick sort buttons function
- [ ] Search box filters users
- [ ] Clicking user loads dashboard
- [ ] Arrow keys navigate users
- [ ] Missing dashboards handled gracefully
- [ ] Responsive design works

### Rollout Plan
1. **Phase 1**: Update visualization.py with new IndexVisualizer
2. **Phase 2**: Test with existing data files
3. **Phase 3**: Verify backward compatibility
4. **Phase 4**: Update documentation

## Risks & Mitigations

### Risk 1: Breaking Existing Functionality
- **Mitigation**: Keep backup of current implementation
- **Recovery**: Revert if issues found

### Risk 2: Performance with Many Users
- **Mitigation**: Test with large datasets
- **Recovery**: Add pagination if needed

### Risk 3: Browser Compatibility
- **Mitigation**: Test in Chrome, Firefox, Safari
- **Recovery**: Add polyfills if needed

## Acceptance Criteria
1. ✓ Index.html generates with user list sidebar
2. ✓ Users can be sorted by ID, total, accepted, rejected, rate
3. ✓ Clicking user loads their dashboard in iframe
4. ✓ Arrow keys switch between users
5. ✓ Search box filters users in real-time
6. ✓ Quick sort buttons provide common sorting options
7. ✓ Works offline without server
8. ✓ Handles missing dashboards gracefully

## Out of Scope
- Database integration
- Real-time updates
- User authentication
- Export functionality
- Comparison views
- Batch operations
- Virtual scrolling for 500+ users

## Open Questions
1. Should we preserve the plotly index_dashboard.html as an alternative view?
2. Do we need to support custom dashboard naming patterns?
3. Should the search also filter by date ranges?
4. Is there a maximum user count we should optimize for?

## Review Cycle
### Self-Review Notes
- Verified backup implementation has all required features
- Confirmed integration points with current code
- Checked that all user requirements are addressed
- Ensured backward compatibility is maintained

### Revisions
- Added quick sort buttons based on backup implementation
- Included keyboard navigation hints
- Added performance optimizations (prefetch, lazy loading)
- Clarified file naming patterns for dashboard discovery