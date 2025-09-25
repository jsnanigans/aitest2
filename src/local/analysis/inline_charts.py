"""
Simple inline chart generators for markdown reports.
Creates ASCII-style and simple matplotlib charts for embedding in reports.
"""

import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class InlineChartGenerator:
    """Generate simple inline charts for markdown reports."""

    @staticmethod
    def create_ascii_bar_chart(values: List[float], labels: List[str],
                              max_width: int = 40, show_values: bool = True) -> str:
        """
        Create a simple ASCII bar chart.

        Args:
            values: List of numerical values
            labels: List of labels for each value
            max_width: Maximum bar width in characters
            show_values: Whether to show value at end of bar

        Returns:
            String containing ASCII bar chart
        """
        if not values:
            return ""

        max_value = max(values)
        chart_lines = []

        for label, value in zip(labels, values):
            if max_value > 0:
                bar_width = int((value / max_value) * max_width)
            else:
                bar_width = 0

            bar = "█" * bar_width
            if show_values:
                line = f"{label:15} {bar} {value:.2f}"
            else:
                line = f"{label:15} {bar}"
            chart_lines.append(line)

        return "\n".join(chart_lines)

    @staticmethod
    def create_ascii_line_chart(values: List[float], width: int = 50, height: int = 10) -> str:
        """
        Create a simple ASCII line chart.

        Args:
            values: List of values to plot
            width: Width of chart in characters
            height: Height of chart in lines

        Returns:
            String containing ASCII line chart
        """
        if not values or len(values) < 2:
            return ""

        # Normalize values to fit in the height
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Create empty grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, value in enumerate(values):
            x = int((i / (len(values) - 1)) * (width - 1))
            y = height - 1 - int(((value - min_val) / range_val) * (height - 1))
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = "●"

        # Add axes
        for i in range(height):
            grid[i][0] = "│"
        for i in range(width):
            grid[height - 1][i] = "─" if grid[height - 1][i] == " " else grid[height - 1][i]
        grid[height - 1][0] = "└"

        # Convert grid to string
        lines = []
        for row in grid:
            lines.append("".join(row))

        return "\n".join(lines)

    @staticmethod
    def create_comparison_bars(raw_value: float, filtered_value: float,
                              label: str, max_width: int = 40) -> str:
        """
        Create side-by-side comparison bars.

        Args:
            raw_value: Raw data value
            filtered_value: Filtered data value
            label: Label for the comparison
            max_width: Maximum bar width

        Returns:
            String with comparison bars
        """
        max_val = max(abs(raw_value), abs(filtered_value))
        if max_val == 0:
            return f"{label}: No data"

        raw_width = int((abs(raw_value) / max_val) * max_width)
        filt_width = int((abs(filtered_value) / max_val) * max_width)

        lines = [
            f"{label}:",
            f"  Raw:      {'█' * raw_width} {raw_value:.2f}",
            f"  Filtered: {'█' * filt_width} {filtered_value:.2f}",
            f"  Change:   {filtered_value - raw_value:+.2f} ({((filtered_value - raw_value) / abs(raw_value) * 100) if raw_value != 0 else 0:.1f}%)"
        ]

        return "\n".join(lines)

    @staticmethod
    def create_mini_sparkline(values: List[float], width: int = 20) -> str:
        """
        Create a tiny sparkline using unicode characters.

        Args:
            values: List of values
            width: Width of sparkline

        Returns:
            String sparkline
        """
        if not values:
            return ""

        # Unicode block elements for sparkline
        blocks = " ▁▂▃▄▅▆▇█"

        # Resample values to fit width
        if len(values) > width:
            # Downsample
            indices = np.linspace(0, len(values) - 1, width).astype(int)
            resampled = [values[i] for i in indices]
        else:
            resampled = values

        # Normalize to 0-8 range
        min_val = min(resampled)
        max_val = max(resampled)
        range_val = max_val - min_val if max_val != min_val else 1

        sparkline = ""
        for value in resampled:
            index = int(((value - min_val) / range_val) * 8)
            sparkline += blocks[min(8, max(0, index))]

        return sparkline

    @staticmethod
    def create_simple_plot(x_values: List[float], y_values: List[float],
                          title: str, xlabel: str, ylabel: str,
                          output_path: str, figsize: Tuple[float, float] = (8, 4)) -> str:
        """
        Create a simple matplotlib plot for embedding.

        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_path: Path to save the plot
            figsize: Figure size

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=figsize)
        plt.plot(x_values, y_values, 'o-', linewidth=2, markersize=6)
        plt.title(title, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path


def generate_report_charts(metrics: dict, output_dir: str = "reports/inline_charts") -> dict:
    """
    Generate a collection of inline charts for a report.

    Args:
        metrics: Dictionary of metrics to visualize
        output_dir: Directory to save chart images

    Returns:
        Dictionary of chart strings and paths
    """
    generator = InlineChartGenerator()
    charts = {}

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Example: Success rates comparison
    if 'success_rates' in metrics:
        rates = metrics['success_rates']
        charts['success_rates_ascii'] = generator.create_comparison_bars(
            rates.get('raw_5pct', 0),
            rates.get('filtered_5pct', 0),
            "5% Weight Loss Success Rate"
        )

    # Example: Weight progression sparkline
    if 'weight_progression' in metrics:
        progression = metrics['weight_progression']
        charts['progression_sparkline'] = generator.create_mini_sparkline(
            progression.get('values', [])
        )

    return charts