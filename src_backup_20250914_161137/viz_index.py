import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import html

def extract_user_stats(results_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    user_stats = []
    users = results_data.get("users", {})

    for user_id, measurements in users.items():
        if not measurements:
            continue

        total = len(measurements)
        accepted = sum(1 for m in measurements if m.get("accepted", False))
        rejected = total - accepted

        dates = [m.get("timestamp") for m in measurements if m.get("timestamp")]
        first_date = min(dates) if dates else None
        last_date = max(dates) if dates else None

        user_stats.append({
            "id": user_id,
            "stats": {
                "total": total,
                "accepted": accepted,
                "rejected": rejected,
                "first_date": str(first_date) if first_date else None,
                "last_date": str(last_date) if last_date else None,
                "acceptance_rate": accepted / total if total > 0 else 0
            }
        })

    return user_stats

def find_dashboard_files(output_dir: str, user_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_path = Path(output_dir)

    for user in user_stats:
        user_id = user["id"]

        possible_files = [
            f"{user_id}.html",  # New simplified dashboard format
            f"dashboard_enhanced_{user_id}.html",
            f"dashboard_{user_id}.html",
            f"viz_{user_id}.html"
        ]

        dashboard_file = None
        for filename in possible_files:
            if (output_path / filename).exists():
                dashboard_file = filename
                break

        user["dashboard_file"] = dashboard_file

    return user_stats

def generate_summary_stats(user_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_users = len(user_stats)
    total_measurements = sum(u["stats"]["total"] for u in user_stats)
    total_accepted = sum(u["stats"]["accepted"] for u in user_stats)

    return {
        "total_users": total_users,
        "total_measurements": total_measurements,
        "total_accepted": total_accepted,
        "total_rejected": total_measurements - total_accepted,
        "overall_acceptance_rate": total_accepted / total_measurements if total_measurements > 0 else 0
    }

def generate_css() -> str:
    return """
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        background: #f5f5f5;
        height: 100vh;
        overflow: hidden;
    }

    .container {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }

    .header {
        background: white;
        border-bottom: 2px solid #e0e0e0;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .header h1 {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .summary-stats {
        display: flex;
        gap: 2rem;
        font-size: 0.9rem;
        color: #666;
    }

    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stat-value {
        font-weight: 600;
        color: #333;
    }

    .main-content {
        display: flex;
        flex: 1;
        overflow: hidden;
    }

    .sidebar {
        width: 320px;
        background: white;
        border-right: 1px solid #e0e0e0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .sidebar-header {
        padding: 1rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .sort-control {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sort-label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
        margin-right: 0.25rem;
    }
    
    .sort-select {
        flex: 1;
        padding: 0.5rem;
        border: 2px solid #e0e0e0;
        border-radius: 6px;
        font-size: 0.9rem;
        background: white;
        cursor: pointer;
        transition: border-color 0.2s;
        font-weight: 500;
    }
    
    .sort-select:hover {
        border-color: #2196f3;
    }
    
    .sort-select:focus {
        outline: none;
        border-color: #2196f3;
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
    }
    
    .sort-direction {
        padding: 0.5rem 0.8rem;
        border: 2px solid #e0e0e0;
        border-radius: 6px;
        background: white;
        cursor: pointer;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.2s;
        color: #2196f3;
    }
    
    .sort-direction:hover {
        background: #f5f5f5;
        border-color: #2196f3;
    }
    
    .sort-direction:active {
        transform: scale(0.95);
    }

    .search-box {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    
    .quick-sort-buttons {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
    }
    
    .quick-sort-btn {
        flex: 1;
        min-width: 80px;
        padding: 0.4rem 0.6rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background: white;
        color: #555;
        font-size: 0.75rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
    }
    
    .quick-sort-btn:hover {
        background: #e3f2fd;
        border-color: #2196f3;
        color: #1976d2;
    }
    
    .quick-sort-btn:active {
        transform: scale(0.95);
    }
    
    .quick-sort-btn.active {
        background: #2196f3;
        color: white;
        border-color: #1976d2;
    }

    .user-list {
        flex: 1;
        overflow-y: auto;
        padding: 0.5rem;
        will-change: scroll-position;
    }

    .user-item {
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
        background: white;
        will-change: auto;
    }

    .user-item:hover {
        background: #f8f9fa;
        border-color: #007bff;
    }

    .user-item.selected {
        background: #e3f2fd;
        border-color: #2196f3;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
    }

    .user-id {
        font-size: 0.85rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .user-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.8rem;
    }

    .user-stat {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .stat-label {
        color: #666;
    }

    .stat-badge {
        padding: 0.15rem 0.4rem;
        border-radius: 12px;
        font-weight: 600;
    }

    .badge-total {
        background: #e0e0e0;
        color: #333;
    }

    .badge-accepted {
        background: #d4edda;
        color: #155724;
    }

    .badge-rejected {
        background: #f8d7da;
        color: #721c24;
    }
    
    .stat-sorted {
        position: relative;
        font-weight: 600;
    }
    
    .stat-sorted .stat-badge {
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
        font-weight: 700;
    }
    
    .user-id.stat-sorted {
        color: #2196f3;
        font-weight: 600;
    }

    .dashboard-container {
        flex: 1;
        padding: 1rem;
        background: #f5f5f5;
        position: relative;
    }

    .dashboard-frame {
        width: 100%;
        height: 100%;
        border: none;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        display: block;
    }

    .no-selection {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #999;
        font-size: 1.1rem;
    }

    .loading {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #666;
    }

    .error-message {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #d32f2f;
        font-size: 1rem;
    }

    .keyboard-hint {
        position: absolute;
        bottom: 1rem;
        right: 1rem;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .keyboard-hint.show {
        opacity: 1;
    }

    @media (max-width: 768px) {
        .sidebar {
            width: 100%;
            position: absolute;
            z-index: 10;
            transition: transform 0.3s;
        }

        .sidebar.hidden {
            transform: translateX(-100%);
        }

        .mobile-toggle {
            display: block;
            position: absolute;
            top: 1rem;
            right: 1rem;
            z-index: 11;
        }
    }

    @media (min-width: 769px) {
        .mobile-toggle {
            display: none;
        }
    }

    .spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    """

def generate_javascript() -> str:
    return """
    class DashboardViewer {
        constructor(data) {
            this.data = data;
            this.currentUser = null;
            this.sortField = 'id';
            this.sortDirection = 'asc';
            this.filteredUsers = [...data.users];
            this.searchTimer = null;
            this.loadingIframe = false;
            this.init();
        }

        init() {
            this.setupEventListeners();
            this.renderUserList();
            this.showKeyboardHint();

            if (this.filteredUsers.length > 0) {
                this.selectUser(this.filteredUsers[0]);
            }
        }

        setupEventListeners() {
            document.getElementById('sortSelect').addEventListener('change', (e) => {
                this.sortField = e.target.value;
                this.sortUsers();
                // Clear active quick sort button
                document.querySelectorAll('.quick-sort-btn').forEach(b => {
                    b.classList.remove('active');
                });
            });

            document.getElementById('sortDirection').addEventListener('click', () => {
                this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
                document.getElementById('sortDirection').textContent =
                    this.sortDirection === 'asc' ? '↑' : '↓';
                this.sortUsers();
            });

            document.getElementById('searchBox').addEventListener('input', (e) => {
                clearTimeout(this.searchTimer);
                this.searchTimer = setTimeout(() => {
                    this.filterUsers(e.target.value);
                }, 150);
            });

            document.addEventListener('keydown', (e) => {
                this.handleKeyboard(e);
            });
            
            // Quick sort buttons
            document.querySelectorAll('.quick-sort-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const sortField = e.target.dataset.sort;
                    const sortDirection = e.target.dataset.direction;
                    
                    // Update UI
                    document.getElementById('sortSelect').value = sortField;
                    this.sortField = sortField;
                    this.sortDirection = sortDirection;
                    document.getElementById('sortDirection').textContent = 
                        sortDirection === 'asc' ? '↑' : '↓';
                    
                    // Update active button
                    document.querySelectorAll('.quick-sort-btn').forEach(b => {
                        b.classList.remove('active');
                    });
                    e.target.classList.add('active');
                    
                    this.sortUsers();
                });
            });
        }

        filterUsers(searchTerm) {
            if (!searchTerm) {
                this.filteredUsers = [...this.data.users];
            } else {
                const term = searchTerm.toLowerCase();
                this.filteredUsers = this.data.users.filter(user =>
                    user.id.toLowerCase().includes(term)
                );
            }
            this.sortUsers();
        }

        sortUsers() {
            this.filteredUsers.sort((a, b) => {
                let aVal, bVal;

                if (this.sortField === 'id') {
                    aVal = a.id;
                    bVal = b.id;
                } else if (this.sortField === 'total') {
                    aVal = a.stats.total;
                    bVal = b.stats.total;
                } else if (this.sortField === 'accepted') {
                    aVal = a.stats.accepted;
                    bVal = b.stats.accepted;
                } else if (this.sortField === 'rejected') {
                    aVal = a.stats.rejected;
                    bVal = b.stats.rejected;
                } else if (this.sortField === 'rate') {
                    aVal = a.stats.acceptance_rate;
                    bVal = b.stats.acceptance_rate;
                } else if (this.sortField === 'first_date') {
                    aVal = a.stats.first_date || '';
                    bVal = b.stats.first_date || '';
                } else if (this.sortField === 'last_date') {
                    aVal = a.stats.last_date || '';
                    bVal = b.stats.last_date || '';
                }

                if (aVal === null || aVal === undefined) aVal = '';
                if (bVal === null || bVal === undefined) bVal = '';

                if (typeof aVal === 'string') {
                    return this.sortDirection === 'asc'
                        ? aVal.localeCompare(bVal)
                        : bVal.localeCompare(aVal);
                } else {
                    return this.sortDirection === 'asc'
                        ? aVal - bVal
                        : bVal - aVal;
                }
            });

            this.renderUserList();

            if (this.currentUser && !this.filteredUsers.includes(this.currentUser)) {
                if (this.filteredUsers.length > 0) {
                    this.selectUser(this.filteredUsers[0]);
                } else {
                    this.currentUser = null;
                    this.showNoSelection();
                }
            }
        }

        renderUserList() {
            const container = document.getElementById('userList');

            if (this.filteredUsers.length === 0) {
                container.innerHTML = '<div style="padding: 1rem; text-align: center; color: #999;">No users found</div>';
                return;
            }

            const existingItems = container.querySelectorAll('.user-item');
            const existingMap = new Map();
            existingItems.forEach(item => {
                existingMap.set(item.dataset.userId, item);
            });

            container.innerHTML = '';

            this.filteredUsers.forEach(user => {
                let item = existingMap.get(user.id);

                if (!item) {
                    item = document.createElement('div');
                    item.className = 'user-item';
                    item.dataset.userId = user.id;

                const acceptanceRate = (user.stats.acceptance_rate * 100).toFixed(1);
                
                // Highlight the sorted field
                const highlightClass = (field) => {
                    return this.sortField === field ? 'stat-sorted' : '';
                };
                
                item.innerHTML = `
                    <div class="user-id" title="${user.id}" class="${highlightClass('id')}">${user.id}</div>
                    <div class="user-stats">
                        <div class="user-stat ${highlightClass('total')}">
                            <span class="stat-label">Total:</span>
                            <span class="stat-badge badge-total">${user.stats.total}</span>
                        </div>
                        <div class="user-stat ${highlightClass('accepted')}">
                            <span class="stat-label">✓</span>
                            <span class="stat-badge badge-accepted">${user.stats.accepted}</span>
                        </div>
                        <div class="user-stat ${highlightClass('rejected')}">
                            <span class="stat-label">✗</span>
                            <span class="stat-badge badge-rejected">${user.stats.rejected}</span>
                        </div>
                        <div class="user-stat ${highlightClass('rate')}">
                            <span class="stat-label">${acceptanceRate}%</span>
                        </div>
                    </div>
                `;

                    item.addEventListener('click', () => this.selectUser(user));
                }

                if (this.currentUser && this.currentUser.id === user.id) {
                    item.classList.add('selected');
                } else {
                    item.classList.remove('selected');
                }

                container.appendChild(item);
            });

            if (this.currentUser) {
                requestAnimationFrame(() => {
                    this.scrollToUser(this.currentUser);
                });
            }
        }

        selectUser(user) {
            if (this.currentUser === user) return;

            this.currentUser = user;

            document.querySelectorAll('.user-item').forEach(item => {
                if (item.dataset.userId === user.id) {
                    item.classList.add('selected');
                } else {
                    item.classList.remove('selected');
                }
            });

            this.loadDashboard(user);
            requestAnimationFrame(() => {
                this.scrollToUser(user);
            });
        }

        loadDashboard(user) {
            if (this.loadingIframe) return;

            const container = document.getElementById('dashboardContainer');

            if (!user.dashboard_file) {
                container.innerHTML = `
                    <div class="error-message">
                        Dashboard not found for user ${user.id}
                    </div>
                `;
                return;
            }

            const existingIframe = container.querySelector('iframe');
            if (existingIframe && existingIframe.src.endsWith(user.dashboard_file)) {
                return;
            }

            this.loadingIframe = true;

            const iframe = document.createElement('iframe');
            iframe.className = 'dashboard-frame';
            iframe.style.visibility = 'hidden';
            iframe.src = user.dashboard_file;

            let loadHandled = false;
            let spinnerTimer = null;

            // Only show spinner if loading takes more than 100ms
            spinnerTimer = setTimeout(() => {
                if (!loadHandled) {
                    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
                }
            }, 100);

            const finishLoading = () => {
                if (!loadHandled) {
                    loadHandled = true;
                    clearTimeout(spinnerTimer);
                    iframe.style.visibility = 'visible';
                    container.innerHTML = '';
                    container.appendChild(iframe);
                    this.loadingIframe = false;
                    this.preloadAdjacent();
                }
            };

            iframe.onload = finishLoading;

            iframe.onerror = () => {
                if (!loadHandled) {
                    loadHandled = true;
                    clearTimeout(spinnerTimer);
                    container.innerHTML = `
                        <div class="error-message">
                            Failed to load dashboard for user ${user.id}
                        </div>
                    `;
                    this.loadingIframe = false;
                }
            };

            // Fallback for edge cases - reduced from 3000ms to 500ms
            setTimeout(() => {
                if (!loadHandled) {
                    finishLoading();
                }
            }, 10);

            // Start loading immediately (hidden)
            container.appendChild(iframe);
        }

        preloadAdjacent() {
            if (!this.currentUser) return;

            const currentIndex = this.filteredUsers.findIndex(u => u.id === this.currentUser.id);
            if (currentIndex === -1) return;

            const preloadUser = (user) => {
                if (!user || !user.dashboard_file) return;
                const link = document.createElement('link');
                link.rel = 'prefetch';
                link.href = user.dashboard_file;
                link.as = 'document';
                document.head.appendChild(link);
            };

            if (currentIndex > 0) {
                preloadUser(this.filteredUsers[currentIndex - 1]);
            }
            if (currentIndex < this.filteredUsers.length - 1) {
                preloadUser(this.filteredUsers[currentIndex + 1]);
            }
        }

        showNoSelection() {
            const container = document.getElementById('dashboardContainer');
            container.innerHTML = `
                <div class="no-selection">
                    Select a user from the list to view their dashboard
                </div>
            `;
        }

        scrollToUser(user) {
            const item = document.querySelector(`[data-user-id="${user.id}"]`);
            if (item) {
                const container = item.parentElement;
                const itemRect = item.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();

                if (itemRect.top < containerRect.top || itemRect.bottom > containerRect.bottom) {
                    item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }
        }

        handleKeyboard(e) {
            if (e.target.tagName === 'INPUT') return;

            const currentIndex = this.filteredUsers.findIndex(u => u.id === this.currentUser?.id);

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (currentIndex < this.filteredUsers.length - 1) {
                    this.selectUser(this.filteredUsers[currentIndex + 1]);
                }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (currentIndex > 0) {
                    this.selectUser(this.filteredUsers[currentIndex - 1]);
                }
            } else if (e.key === 'Home') {
                e.preventDefault();
                if (this.filteredUsers.length > 0) {
                    this.selectUser(this.filteredUsers[0]);
                }
            } else if (e.key === 'End') {
                e.preventDefault();
                if (this.filteredUsers.length > 0) {
                    this.selectUser(this.filteredUsers[this.filteredUsers.length - 1]);
                }
            }
        }

        showKeyboardHint() {
            const hint = document.getElementById('keyboardHint');
            if (hint) {
                hint.classList.add('show');
                setTimeout(() => {
                    hint.classList.remove('show');
                }, 3000);
            }
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        if (typeof DASHBOARD_DATA !== 'undefined') {
            window.viewer = new DashboardViewer(DASHBOARD_DATA);
        }
    });
    """

def generate_index_html(
    user_stats: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: str,
    generated_time: Optional[str] = None
) -> str:

    if not generated_time:
        generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "generated": generated_time,
        "output_dir": os.path.basename(output_dir),
        "users": user_stats,
        "summary": summary
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weight Stream Processor - Dashboard Viewer</title>
    <link rel="preload" href="plotly.min.js" as="script">
    <style>
{generate_css()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Weight Stream Processor Dashboard Viewer</h1>
            <div class="summary-stats">
                <div class="stat-item">
                    <span>Users:</span>
                    <span class="stat-value">{summary['total_users']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Measurements:</span>
                    <span class="stat-value">{summary['total_measurements']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Accepted:</span>
                    <span class="stat-value" style="color: #28a745;">{summary['total_accepted']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Rejected:</span>
                    <span class="stat-value" style="color: #dc3545;">{summary['total_rejected']:,}</span>
                </div>
                <div class="stat-item">
                    <span>Acceptance Rate:</span>
                    <span class="stat-value">{summary['overall_acceptance_rate']:.1%}</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">
                    <div class="sort-control">
                        <span class="sort-label">Sort by:</span>
                        <select id="sortSelect" class="sort-select">
                            <option value="id">User ID</option>
                            <option value="total">Total Measurements</option>
                            <option value="accepted">Accepted Count</option>
                            <option value="rejected">Rejected Count</option>
                            <option value="rate">Acceptance Rate (%)</option>
                            <option value="first_date">First Date</option>
                            <option value="last_date">Last Date</option>
                        </select>
                        <button id="sortDirection" class="sort-direction" title="Click to reverse sort">↑</button>
                    </div>
                    <input type="text" id="searchBox" class="search-box" placeholder="Search users...">
                    <div class="quick-sort-buttons">
                        <button class="quick-sort-btn" data-sort="rejected" data-direction="desc" title="Most rejected first">
                            Most Rejected
                        </button>
                        <button class="quick-sort-btn" data-sort="rate" data-direction="asc" title="Lowest acceptance rate first">
                            Lowest Rate
                        </button>
                        <button class="quick-sort-btn" data-sort="total" data-direction="desc" title="Most measurements first">
                            Most Data
                        </button>
                    </div>
                </div>
                <div class="user-list" id="userList">
                </div>
            </div>

            <div class="dashboard-container" id="dashboardContainer">
                <div class="no-selection">
                    Select a user from the list to view their dashboard
                </div>
            </div>

            <div class="keyboard-hint" id="keyboardHint">
                Use ↑↓ arrow keys to navigate • Home/End to jump
            </div>
        </div>
    </div>

    <script>
    const DASHBOARD_DATA = {json.dumps(data)};
    </script>
    <script>
{generate_javascript()}
    </script>
</body>
</html>"""

    return html_content

def create_index_from_results(
    results_file: str,
    output_dir: str,
    output_filename: str = "index.html"
) -> str:
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_path, 'r') as f:
        results_data = json.load(f)

    user_stats = extract_user_stats(results_data)
    user_stats = find_dashboard_files(output_dir, user_stats)
    summary = generate_summary_stats(user_stats)

    html_content = generate_index_html(
        user_stats,
        summary,
        output_dir,
        results_data.get("stats", {}).get("start_time")
    )

    output_path = Path(output_dir) / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(output_path)
