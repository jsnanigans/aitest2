# Plan: Interactive Index Dashboard Viewer

## Summary
Create an index.html page that serves as a centralized dashboard viewer for all user weight stream processing reports. The page will feature a sortable user list sidebar with statistics and an iframe-based report viewer with keyboard navigation support.

## Context
- Source: User description
- Assumptions:
  - Reports are already generated as HTML files in output directories
  - Each user has a unique ID and corresponding dashboard HTML file
  - Statistics can be extracted from results JSON files
  - Users want quick navigation between reports without page reloads
- Constraints:
  - Must work with existing dashboard HTML files (enhanced plotly dashboards)
  - Should be standalone (no server required, file:// protocol compatible)
  - Must handle potentially hundreds of users efficiently

## Requirements

### Functional
1. Display list of all processed users in a sidebar
2. Show statistics per user (total values, accepted, rejected counts)
3. Enable sorting by user ID, total values, accepted, or rejected counts
4. Load selected user's dashboard in an iframe
5. Support keyboard navigation (up/down arrows to switch users)
6. Highlight currently selected user in the list
7. Show summary statistics at the top (total users, overall acceptance rate)
8. Auto-select first user on page load
9. Handle missing dashboard files gracefully

### Non-functional
1. Fast loading and responsive UI even with 100+ users
2. Clean, professional appearance matching existing dashboard style
3. Accessible keyboard navigation
4. Works offline (no external dependencies)
5. Responsive layout that adapts to different screen sizes

## Alternatives

### Option A: Static HTML Generation
- Approach: Generate index.html during processing with embedded data
- Pros: 
  - No runtime data loading needed
  - Single file solution
  - Fast initial load
- Cons:
  - Requires regeneration when data changes
  - Less flexible for future enhancements
  - Larger file size with embedded data
- Risks: File becomes too large with many users

### Option B: Dynamic JavaScript Loading
- Approach: Load results.json dynamically and build UI
- Pros:
  - Smaller HTML file
  - Can reload data without regenerating HTML
  - More flexible for filtering/searching
- Cons:
  - Requires results.json to be accessible
  - CORS issues with file:// protocol
  - Slightly slower initial load
- Risks: Browser security restrictions

### Option C: Hybrid Approach (Recommended)
- Approach: Generate HTML with embedded data but modular JavaScript
- Pros:
  - Works offline immediately
  - Can be enhanced to load external data later
  - Optimal performance
  - No CORS issues
- Cons:
  - Slightly more complex implementation
  - HTML file grows with user count
- Risks: Minimal

## Recommendation
**Option C (Hybrid Approach)** - Generate an index.html with embedded user data during processing, but structure the JavaScript to be modular and extensible. This provides the best balance of performance, reliability, and future flexibility.

## High-Level Design

### Architecture
```
index.html
├── Embedded Data (JSON in script tag)
├── CSS (embedded for standalone use)
├── JavaScript Components
│   ├── UserList Manager
│   ├── Statistics Calculator
│   ├── Sort Handler
│   ├── Keyboard Navigation
│   └── IFrame Loader
└── HTML Structure
    ├── Header (summary stats)
    ├── Sidebar (user list)
    └── Main Content (iframe)
```

### Data Model
```javascript
{
  "generated": "timestamp",
  "output_dir": "viz_test_no_date",
  "users": [
    {
      "id": "user-uuid",
      "dashboard_file": "dashboard_enhanced_user-uuid.html",
      "stats": {
        "total": 100,
        "accepted": 95,
        "rejected": 5,
        "first_date": "2022-01-01",
        "last_date": "2023-12-31",
        "acceptance_rate": 0.95
      }
    }
  ],
  "summary": {
    "total_users": 95,
    "total_measurements": 9500,
    "overall_acceptance_rate": 0.94
  }
}
```

### UI Layout
```
┌─────────────────────────────────────────────────┐
│ Weight Stream Processor Dashboard Viewer        │
│ 95 users | 9,500 measurements | 94% accepted    │
├─────────────────┬───────────────────────────────┤
│ Sort by: [▼]    │                               │
│ ┌─────────────┐ │                               │
│ │ User ID     │ │                               │
│ │ Total       │ │   ┌───────────────────────┐  │
│ │ Accepted    │ │   │                       │  │
│ │ Rejected    │ │   │                       │  │
│ └─────────────┘ │   │    Dashboard          │  │
│                 │   │     IFrame            │  │
│ Users:          │   │                       │  │
│ ┌─────────────┐ │   │                       │  │
│ │▶ User1      │ │   │                       │  │
│ │  95/5       │ │   │                       │  │
│ ├─────────────┤ │   │                       │  │
│ │  User2      │ │   │                       │  │
│ │  88/12      │ │   └───────────────────────┘  │
│ └─────────────┘ │                               │
└─────────────────┴───────────────────────────────┘
```

## Implementation Plan (No Code)

### Step 1: Create Index Generator Module
- Location: `src/viz_index.py`
- Purpose: Generate index.html from results data
- Functions:
  - `extract_user_stats()`: Parse results.json for user statistics
  - `find_dashboard_files()`: Match users to dashboard HTML files
  - `generate_index_html()`: Create the complete index.html
  - `embed_data()`: Embed JSON data in script tag
  - `generate_css()`: Create embedded styles
  - `generate_javascript()`: Create UI logic

### Step 2: Integrate with Main Processing
- Modify: `main.py`
- After visualization generation, call index generator
- Pass results data and output directory
- Save index.html in visualization directory

### Step 3: HTML Structure Components
- Header section with summary statistics
- Sidebar with:
  - Sort dropdown
  - User list container
  - User item template (ID, accepted/rejected counts)
- Main content area with iframe
- Hidden data script tag

### Step 4: JavaScript Functionality
- User list rendering and updating
- Sort functionality (by ID, total, accepted, rejected)
- Click handler for user selection
- Keyboard navigation (up/down arrows)
- IFrame source management
- Error handling for missing dashboards
- Responsive layout adjustments

### Step 5: CSS Styling
- Consistent with existing dashboard style
- Sidebar styling (scrollable, fixed width)
- User item styling (hover, selected states)
- Statistics badges (accepted=green, rejected=red)
- Responsive breakpoints
- Loading states

### Step 6: Performance Optimizations
- Virtual scrolling for large user lists (100+ users)
- Lazy loading of dashboard content
- Debounced keyboard navigation
- Efficient DOM updates
- Minimize reflows during sorting

### Step 7: Enhanced Features
- Search/filter box for finding users
- Export selected users list
- Batch operations (open multiple dashboards)
- Remember last selected user (localStorage)
- Fullscreen mode for dashboard viewing

## Validation & Rollout

### Test Strategy
1. Unit tests for stats extraction and HTML generation
2. Test with various user counts (1, 10, 100, 1000 users)
3. Test sorting functionality with edge cases
4. Test keyboard navigation boundaries
5. Test with missing dashboard files
6. Cross-browser testing (Chrome, Firefox, Safari)
7. Test file:// protocol access

### Manual QA Checklist
- [ ] Index.html loads without errors
- [ ] All users appear in sidebar
- [ ] Statistics are accurate
- [ ] Sorting works for all columns
- [ ] Click selection works
- [ ] Keyboard navigation works
- [ ] Dashboard loads in iframe
- [ ] Selected user is highlighted
- [ ] Responsive layout works
- [ ] Performance acceptable with 100+ users

### Rollout Plan
1. Phase 1: Generate basic index.html with core features
2. Phase 2: Add sorting and keyboard navigation
3. Phase 3: Add search and enhanced features
4. Phase 4: Performance optimizations for large datasets

## Risks & Mitigations

### Risk 1: Browser Security Restrictions
- Issue: CORS/security blocks with file:// protocol
- Mitigation: Embed all data, use relative paths
- Fallback: Provide instructions for local server

### Risk 2: Performance with Many Users
- Issue: UI becomes slow with 500+ users
- Mitigation: Implement virtual scrolling
- Fallback: Pagination or lazy loading

### Risk 3: Large HTML File Size
- Issue: Index.html becomes too large with embedded data
- Mitigation: Compress/minimize data structure
- Fallback: External JSON with local server option

## Acceptance Criteria
1. ✓ Index.html displays all processed users
2. ✓ Shows accurate statistics per user
3. ✓ Sorting works on all columns
4. ✓ Click to load dashboard in iframe
5. ✓ Up/down arrows navigate between users
6. ✓ Works offline without external dependencies
7. ✓ Loads in under 2 seconds with 100 users
8. ✓ Responsive layout on different screen sizes

## Out of Scope
- Server-side functionality
- Real-time data updates
- Database integration
- User authentication
- Dashboard editing capabilities
- Comparison views between users
- Data export functionality (beyond basic)

## Open Questions
1. Should we add a search/filter box in the initial version?
2. What's the expected maximum number of users?
3. Should we persist user selection between sessions?
4. Do we need to support printing the index view?
5. Should sorting preferences be remembered?

## Review Cycle
### Self-Review Notes
- Considered council input:
  - **Butler Lampson**: "Keep it simple - embed data for reliability"
  - **Don Norman**: "Focus on keyboard navigation and visual feedback"
  - **Kent Beck**: "Start with minimal viable features, iterate"
- Added virtual scrolling consideration for scalability
- Included fallback strategies for browser restrictions
- Structured for progressive enhancement

### Revisions
- v1.1: Added keyboard navigation details
- v1.2: Included performance optimization section
- v1.3: Added responsive design requirements