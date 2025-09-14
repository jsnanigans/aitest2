"""
Visualization constants for consistent styling across all charts.
Defines colors, markers, and mappings for source types and rejection categories.
"""

from typing import Dict, Tuple

# Source types found in data with their visual properties
SOURCE_TYPE_STYLES: Dict[str, Dict[str, any]] = {
    # Most common sources - use distinct primary colors and shapes
    'patient-device': {
        'color': '#2E7D32',      # Forest green
        'marker': 'o',           # Circle
        'size': 60,
        'label': 'Patient Device',
        'priority': 1
    },
    'https://api.iglucose.com': {
        'color': '#1565C0',      # Deep blue
        'marker': 's',           # Square
        'size': 55,
        'label': 'iGlucose API',
        'priority': 2
    },
    'https://connectivehealth.io': {
        'color': '#6A1B9A',      # Deep purple
        'marker': '^',           # Triangle up
        'size': 65,
        'label': 'ConnectiveHealth',
        'priority': 3
    },
    'patient-upload': {
        'color': '#E65100',      # Deep orange
        'marker': 'D',           # Diamond
        'size': 55,
        'label': 'Patient Upload',
        'priority': 4
    },
    'internal-questionnaire': {
        'color': '#00695C',      # Teal
        'marker': 'p',           # Pentagon
        'size': 60,
        'label': 'Questionnaire',
        'priority': 5
    },
    'care-team-upload': {
        'color': '#AD1457',      # Deep pink
        'marker': 'h',           # Hexagon
        'size': 60,
        'label': 'Care Team',
        'priority': 6
    },
    'solera': {
        'color': '#4E342E',      # Dark brown
        'marker': '*',           # Star
        'size': 70,
        'label': 'Solera',
        'priority': 7
    },
    # Catch-all for unknown sources
    'unknown': {
        'color': '#757575',      # Grey
        'marker': '.',           # Point
        'size': 40,
        'label': 'Unknown',
        'priority': 999
    }
}

# Rejection category colors - high contrast, visually distinct palette
REJECTION_CATEGORY_COLORS: Dict[str, str] = {
    # Most Severe (darkest)
    "BMI Value": "#000000",        # Black
    "Unit Convert": "#FF0000",     # Pure red
    "Physio Limit": "#0000FF",     # Pure blue

    # High Severity
    "Extreme Dev": "#FF00FF",      # Magenta
    "Out of Bounds": "#00FF00",    # Lime green

    # Medium Severity
    "High Variance": "#FF8C00",    # Dark orange
    "Sustained": "#e24b1d",
    "Limit Exceed": "#9400D3",     # Violet

    # Lower Severity (lightest)
    "Daily Flux": "#931648",
    "Medium Term": "#8B4513",      # Saddle brown
    "Short Term": "#4682B4",       # Steel blue

    # Unknown
    "Other": "#808080"             # Grey
}

def get_source_style(source: str) -> Dict[str, any]:
    """Get the visual style for a given source type."""
    # Direct match
    if source in SOURCE_TYPE_STYLES:
        return SOURCE_TYPE_STYLES[source].copy()

    # Partial match for source variations
    source_lower = source.lower() if source else ''

    # Check for iglucose variants
    if 'iglucose' in source_lower:
        return SOURCE_TYPE_STYLES['https://api.iglucose.com'].copy()

    # Check for connectivehealth variants
    if 'connectivehealth' in source_lower:
        return SOURCE_TYPE_STYLES['https://connectivehealth.io'].copy()

    # Check for device-related sources
    if 'device' in source_lower:
        return SOURCE_TYPE_STYLES['patient-device'].copy()

    # Check for upload-related sources
    if 'upload' in source_lower:
        if 'team' in source_lower or 'care' in source_lower:
            return SOURCE_TYPE_STYLES['care-team-upload'].copy()
        return SOURCE_TYPE_STYLES['patient-upload'].copy()

    # Check for questionnaire variants
    if 'questionnaire' in source_lower:
        return SOURCE_TYPE_STYLES['internal-questionnaire'].copy()

    # Default to unknown
    return SOURCE_TYPE_STYLES['unknown'].copy()

def get_rejection_color(category: str) -> str:
    """Get the color for a given rejection category."""
    return REJECTION_CATEGORY_COLORS.get(category, REJECTION_CATEGORY_COLORS['Other'])

# Export all source types for reference
ALL_SOURCE_TYPES = list(SOURCE_TYPE_STYLES.keys())
ALL_REJECTION_CATEGORIES = list(REJECTION_CATEGORY_COLORS.keys())
