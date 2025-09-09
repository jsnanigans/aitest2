#!/usr/bin/env python3
"""Create a reference legend showing all source type icons/markers."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Define the same markers and colors used in the visualization
source_styles = {
    'care-team-upload': {'marker': '^', 'color': '#2E7D32', 'label': 'Care Team Upload'},
    'patient-device': {'marker': 's', 'color': '#1565C0', 'label': 'Patient Device'}, 
    'internal-questionnaire': {'marker': 'D', 'color': '#7B1FA2', 'label': 'Internal Questionnaire'},
    'patient-upload': {'marker': 'v', 'color': '#E65100', 'label': 'Patient Upload'},
    'https://connectivehealth.io': {'marker': 'p', 'color': '#00695C', 'label': 'Connective Health'},
    'https://api.iglucose.com': {'marker': 'h', 'color': '#B71C1C', 'label': 'iGlucose API'},
    'unknown': {'marker': 'o', 'color': '#616161', 'label': 'Unknown Source'}
}

# Create figure for legend
fig, ax = plt.subplots(figsize=(10, 6))

# Hide the axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(5, 7.5, 'Weight Data Source Type Indicators', 
        fontsize=16, fontweight='bold', ha='center')

# Create a visual grid showing each source type
y_pos = 6.5
for i, (source, style) in enumerate(source_styles.items()):
    if i % 2 == 0:
        x_pos = 2
        if i > 0:
            y_pos -= 1.2
    else:
        x_pos = 6
    
    # Draw the marker
    ax.scatter(x_pos - 0.5, y_pos, 
              marker=style['marker'], 
              c=style['color'], 
              s=200, 
              alpha=0.9,
              edgecolors='white', 
              linewidth=1.0)
    
    # Add label
    ax.text(x_pos, y_pos, style['label'], 
           fontsize=12, va='center')
    
    # Add source key in smaller text
    ax.text(x_pos, y_pos - 0.25, f'({source})', 
           fontsize=9, va='center', style='italic', alpha=0.7)

# Add explanation text
explanation = """
Each data point in the weight tracking graphs uses a different marker shape and color
to indicate its source. This helps identify where each weight measurement came from.

Size and opacity indicate confidence level:
• Large/Solid = High confidence (≥0.75)
• Medium = Medium confidence (0.5-0.75)  
• Small/Faint = Low confidence (<0.5)

Predicted values appear as hollow markers with dashed outlines.
"""

ax.text(5, 1.5, explanation, fontsize=10, ha='center', va='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save the legend
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'source_type_legend.png'
plt.savefig(output_file, dpi=120, bbox_inches='tight')
plt.close()

print(f"Created source type legend: {output_file}")
print("\nMarker shapes used for each source:")
print("-" * 40)
for source, style in source_styles.items():
    marker_desc = {
        '^': 'Triangle Up',
        's': 'Square',
        'D': 'Diamond',
        'v': 'Triangle Down',
        'p': 'Pentagon',
        'h': 'Hexagon',
        'o': 'Circle'
    }
    print(f"{style['label']:25s} : {marker_desc.get(style['marker'], style['marker'])} ({style['color']})")