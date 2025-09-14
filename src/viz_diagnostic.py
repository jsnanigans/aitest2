import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

class DiagnosticDashboard:
    """
    Enhanced diagnostic dashboard for weight processing analysis.
    Focuses on explaining every decision and showing complete quality metrics.
    """
    
    def create_diagnostic_dashboard(
        self,
        results: List[Dict[str, Any]],
        user_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create comprehensive diagnostic dashboard with full transparency."""
        
        if not results:
            return self._create_empty_dashboard(user_id)
        
        # Create figure with specific subplot layout
        fig = make_subplots(
            rows=4, cols=3,
            row_heights=[0.35, 0.25, 0.20, 0.20],
            column_widths=[0.4, 0.3, 0.3],
            subplot_titles=(
                "Weight Processing Timeline with Decisions",
                "Quality Score Breakdown",
                "Rejection Analysis",
                "Kalman Innovation & Residuals",
                "Normalized Innovation Distribution", 
                "Confidence Evolution",
                "Component Score Timeline",
                "Gap Reset Events",
                "Source Reliability Impact"
            ),
            specs=[
                [{"colspan": 3}, None, None],  # Main timeline spans full width
                [{"type": "scatter"}, {"type": "bar"}, {"type": "domain"}],  # sunburst is domain type
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.10
        )
        
        # Process data
        df = self._prepare_dataframe(results)
        
        # 1. Main Timeline with Complete Information
        self._add_main_timeline(fig, df, row=1, col=1)
        
        # 2. Quality Score Breakdown (Stacked Area)
        self._add_quality_breakdown(fig, df, row=2, col=1)
        
        # 3. Rejection Reason Bar Chart
        self._add_rejection_analysis(fig, df, row=2, col=2)
        
        # 4. Rejection Sunburst
        self._add_rejection_sunburst(fig, df, row=2, col=3)
        
        # 5. Kalman Diagnostics
        self._add_kalman_diagnostics(fig, df, row=3, col=1)
        
        # 6. Innovation Distribution
        self._add_innovation_distribution(fig, df, row=3, col=2)
        
        # 7. Confidence Evolution
        self._add_confidence_evolution(fig, df, row=3, col=3)
        
        # 8. Component Scores Over Time
        self._add_component_timeline(fig, df, row=4, col=1)
        
        # 9. Gap Reset Table
        self._add_gap_reset_table(fig, df, row=4, col=2)
        
        # 10. Source Impact Analysis
        self._add_source_impact(fig, df, row=4, col=3)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Weight Processing Diagnostic Dashboard - User {user_id[:8]}",
                font=dict(size=20, color='#1565C0'),
                x=0.5,
                xanchor='center'
            ),
            height=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_empty_dashboard(self, user_id: str) -> go.Figure:
        """Create empty dashboard when no data available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title=f"Weight Processing Diagnostic Dashboard - User {user_id[:8]}",
            height=600
        )
        return fig
    
    def _add_main_timeline(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add main timeline with all decision points and annotations."""
        
        # Accepted measurements with quality score color coding
        accepted_df = df[df['accepted'] == True].copy()
        if not accepted_df.empty:
            # Add Kalman filtered line
            if 'filtered_weight' in accepted_df.columns and not accepted_df['filtered_weight'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=accepted_df['timestamp'],
                        y=accepted_df['filtered_weight'],
                        mode='lines',
                        name='Kalman Filtered',
                        line=dict(color='#1565C0', width=2),
                        hovertemplate=(
                            '<b>Kalman Filtered</b><br>' +
                            'Weight: %{y:.2f} kg<br>' +
                            'Time: %{x}<br>' +
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )
                
                # Add uncertainty bands if we have innovation data
                if 'innovation' in accepted_df.columns and not accepted_df['innovation'].isna().all():
                    uncertainty = accepted_df['innovation'].abs() * 0.5
                    uncertainty = uncertainty.fillna(0.5)
                    upper_band = accepted_df['filtered_weight'] + uncertainty
                    lower_band = accepted_df['filtered_weight'] - uncertainty
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([accepted_df['timestamp'], accepted_df['timestamp'][::-1]]),
                            y=pd.concat([upper_band, lower_band[::-1]]),
                            fill='toself',
                            fillcolor='rgba(21, 101, 192, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Uncertainty Band',
                            showlegend=True,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
            
            # Prepare custom data for hover
            hover_data = []
            for col_name in ['quality_score', 'source', 'innovation', 'normalized_innovation', 
                           'safety', 'plausibility', 'consistency', 'reliability']:
                if col_name in accepted_df.columns:
                    hover_data.append(accepted_df[col_name].fillna(0))
                else:
                    hover_data.append([0] * len(accepted_df))
            
            # Add accepted points with quality color coding
            colors = accepted_df['quality_score'].fillna(0.5) if 'quality_score' in accepted_df.columns else [0.5] * len(accepted_df)
            
            fig.add_trace(
                go.Scatter(
                    x=accepted_df['timestamp'],
                    y=accepted_df['raw_weight'],
                    mode='markers',
                    name='Accepted',
                    marker=dict(
                        size=10,
                        color=colors,
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title='Quality<br>Score',
                            thickness=15,
                            len=0.3,
                            y=0.85,
                            yanchor='top'
                        ),
                        line=dict(color='white', width=1)
                    ),
                    customdata=np.column_stack(hover_data) if hover_data else None,
                    hovertemplate=(
                        '<b>Accepted Measurement</b><br>' +
                        'Raw Weight: %{y:.2f} kg<br>' +
                        'Quality Score: %{customdata[0]:.2f}<br>' +
                        'Source: %{customdata[1]}<br>' +
                        'Innovation: %{customdata[2]:.3f} kg<br>' +
                        'Norm. Innovation: %{customdata[3]:.2f}σ<br>' +
                        '<b>Quality Components:</b><br>' +
                        '  Safety: %{customdata[4]:.2f}<br>' +
                        '  Plausibility: %{customdata[5]:.2f}<br>' +
                        '  Consistency: %{customdata[6]:.2f}<br>' +
                        '  Reliability: %{customdata[7]:.2f}<br>' +
                        '<extra></extra>'
                    ) if hover_data else None
                ),
                row=row, col=col
            )
        
        # Rejected measurements with detailed reasons
        rejected_df = df[df['accepted'] == False].copy()
        if not rejected_df.empty:
            # Group by rejection category for different markers
            categories = rejected_df['rejection_category'].unique() if 'rejection_category' in rejected_df.columns else ['Unknown']
            
            for category in categories:
                if pd.isna(category):
                    category = 'Unknown'
                cat_df = rejected_df[rejected_df['rejection_category'] == category] if 'rejection_category' in rejected_df.columns else rejected_df
                
                # Prepare hover data
                hover_data = []
                for col_name in ['rejection_reason', 'quality_score', 'source', 'stage']:
                    if col_name in cat_df.columns:
                        hover_data.append(cat_df[col_name].fillna('N/A'))
                    else:
                        hover_data.append(['N/A'] * len(cat_df))
                
                fig.add_trace(
                    go.Scatter(
                        x=cat_df['timestamp'],
                        y=cat_df['raw_weight'],
                        mode='markers',
                        name=f'Rejected: {category}',
                        marker=dict(
                            size=8,
                            symbol='x',
                            color=self._get_rejection_color(category),
                            line=dict(color='darkred', width=1)
                        ),
                        customdata=np.column_stack(hover_data) if hover_data else None,
                        hovertemplate=(
                            '<b>Rejected Measurement</b><br>' +
                            'Raw Weight: %{y:.2f} kg<br>' +
                            'Reason: %{customdata[0]}<br>' +
                            'Quality Score: %{customdata[1]}<br>' +
                            'Source: %{customdata[2]}<br>' +
                            'Stage: %{customdata[3]}<br>' +
                            '<extra></extra>'
                        ) if hover_data else None
                    ),
                    row=row, col=col
                )
        
        # Add gap reset indicators as scatter markers with text
        reset_df = df[df['was_reset'] == True] if 'was_reset' in df.columns else pd.DataFrame()
        if not reset_df.empty:
            # Get y-axis range for positioning
            all_weights = pd.concat([
                accepted_df['raw_weight'] if not accepted_df.empty else pd.Series(),
                rejected_df['raw_weight'] if not rejected_df.empty else pd.Series()
            ])
            if not all_weights.empty:
                y_max = all_weights.max()
                y_min = all_weights.min()
                y_range = y_max - y_min
                
                # Add vertical lines as shapes (works better with subplots)
                for _, reset in reset_df.iterrows():
                    gap_days = reset['gap_days'] if 'gap_days' in reset else 0
                    
                    # Add a vertical line marker
                    fig.add_trace(
                        go.Scatter(
                            x=[reset['timestamp'], reset['timestamp']],
                            y=[y_min - y_range * 0.1, y_max + y_range * 0.1],
                            mode='lines+text',
                            line=dict(color='gray', width=2, dash='dash'),
                            text=['', f"Reset: {gap_days:.0f}d gap"],
                            textposition='top center',
                            textfont=dict(size=10, color='gray'),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)
    
    def _add_quality_breakdown(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add stacked area chart showing quality component evolution."""
        
        quality_df = df[df['quality_score'].notna()] if 'quality_score' in df.columns else pd.DataFrame()
        if quality_df.empty:
            self._add_no_data_annotation(fig, row, col, "No quality score data available")
            return
        
        components = ['safety', 'plausibility', 'consistency', 'reliability']
        colors = ['#2E7D32', '#1976D2', '#7B1FA2', '#F57C00']
        
        for i, component in enumerate(components):
            if component in quality_df.columns and not quality_df[component].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=quality_df['timestamp'],
                        y=quality_df[component],
                        mode='lines',
                        name=component.capitalize(),
                        stackgroup='quality',
                        fillcolor=colors[i],
                        line=dict(width=0.5, color=colors[i]),
                        hovertemplate=(
                            f'<b>{component.capitalize()}</b><br>' +
                            'Score: %{y:.2f}<br>' +
                            'Time: %{x}<br>' +
                            '<extra></extra>'
                        )
                    ),
                    row=row, col=col
                )
        
        # Add threshold line
        fig.add_hline(
            y=0.6,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text="Threshold",
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Component Score", range=[0, 1], row=row, col=col)
    
    def _add_rejection_analysis(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add rejection reason analysis bar chart."""
        
        rejected_df = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
        if rejected_df.empty or 'rejection_category' not in rejected_df.columns:
            self._add_no_data_annotation(fig, row, col, "No rejections to analyze")
            return
        
        # Count rejections by category
        rejection_counts = rejected_df['rejection_category'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=rejection_counts.values,
                y=rejection_counts.index,
                orientation='h',
                marker=dict(
                    color=[self._get_rejection_color(cat) for cat in rejection_counts.index]
                ),
                text=rejection_counts.values,
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Count", row=row, col=col)
        fig.update_yaxes(title_text="Rejection Category", row=row, col=col)
    
    def _add_rejection_sunburst(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add rejection sunburst chart for hierarchical view."""
        
        rejected_df = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
        if rejected_df.empty or 'rejection_category' not in rejected_df.columns:
            # Can't add annotation to domain type subplot, just add empty sunburst
            fig.add_trace(
                go.Sunburst(
                    labels=['No Data'],
                    parents=[''],
                    values=[1],
                    marker=dict(colors=['lightgray']),
                    hovertemplate='No rejections to visualize<extra></extra>'
                ),
                row=row, col=col
            )
            return
        
        # Prepare hierarchical data
        labels = ['All Rejections']
        parents = ['']
        values = [len(rejected_df)]
        colors = ['lightgray']
        
        # Add categories
        for category in rejected_df['rejection_category'].unique():
            if pd.isna(category):
                continue
            cat_df = rejected_df[rejected_df['rejection_category'] == category]
            labels.append(category)
            parents.append('All Rejections')
            values.append(len(cat_df))
            colors.append(self._get_rejection_color(category))
            
            # Add stages within category if available
            if 'stage' in cat_df.columns:
                for stage in cat_df['stage'].unique():
                    if pd.isna(stage):
                        continue
                    stage_df = cat_df[cat_df['stage'] == stage]
                    labels.append(f"{category} - {stage}")
                    parents.append(category)
                    values.append(len(stage_df))
                    colors.append(self._get_rejection_color(category))
        
        fig.add_trace(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(colors=colors),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_kalman_diagnostics(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add Kalman filter diagnostic plots."""
        
        kalman_df = df[df['innovation'].notna()] if 'innovation' in df.columns else pd.DataFrame()
        if kalman_df.empty:
            self._add_no_data_annotation(fig, row, col, "No Kalman innovation data")
            return
        
        # Innovation (residuals) with color coding
        innovations = kalman_df['innovation'].values
        colors = ['green' if abs(inn) < 1 else 'orange' if abs(inn) < 2 else 'red' 
                  for inn in innovations]
        
        # Prepare hover data
        hover_data = []
        for col_name in ['normalized_innovation', 'confidence']:
            if col_name in kalman_df.columns:
                hover_data.append(kalman_df[col_name].fillna(0))
            else:
                hover_data.append([0] * len(kalman_df))
        
        fig.add_trace(
            go.Scatter(
                x=kalman_df['timestamp'],
                y=kalman_df['innovation'],
                mode='markers+lines',
                name='Innovation',
                marker=dict(
                    size=6,
                    color=colors,
                    line=dict(width=0.5, color='white')
                ),
                line=dict(width=1, color='gray'),
                customdata=np.column_stack(hover_data) if hover_data else None,
                hovertemplate=(
                    '<b>Kalman Innovation</b><br>' +
                    'Innovation: %{y:.3f} kg<br>' +
                    'Normalized: %{customdata[0]:.2f}σ<br>' +
                    'Confidence: %{customdata[1]:.2f}<br>' +
                    '<extra></extra>'
                ) if hover_data else None
            ),
            row=row, col=col
        )
        
        # Add zero line
        fig.add_hline(y=0, line=dict(color='black', width=1), row=row, col=col)
        
        # Add standard deviation bands
        std_innovation = kalman_df['innovation'].std()
        if not pd.isna(std_innovation) and std_innovation > 0:
            fig.add_hrect(
                y0=-std_innovation, y1=std_innovation,
                fillcolor='green', opacity=0.1,
                line_width=0,
                row=row, col=col
            )
            fig.add_hrect(
                y0=-2*std_innovation, y1=2*std_innovation,
                fillcolor='orange', opacity=0.1,
                line_width=0,
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Innovation (kg)", row=row, col=col)
    
    def _add_innovation_distribution(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add normalized innovation distribution histogram."""
        
        norm_inn_df = df[df['normalized_innovation'].notna()] if 'normalized_innovation' in df.columns else pd.DataFrame()
        if norm_inn_df.empty:
            self._add_no_data_annotation(fig, row, col, "No normalized innovation data")
            return
        
        fig.add_trace(
            go.Histogram(
                x=norm_inn_df['normalized_innovation'],
                nbinsx=20,
                marker=dict(
                    color='#1976D2',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Normalized Innovation</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add reference lines
        fig.add_vline(x=0, line=dict(color='black', width=2), row=row, col=col)
        fig.add_vline(x=2, line=dict(color='orange', width=1, dash='dash'), row=row, col=col)
        fig.add_vline(x=-2, line=dict(color='orange', width=1, dash='dash'), row=row, col=col)
        fig.add_vline(x=3, line=dict(color='red', width=1, dash='dash'), row=row, col=col)
        fig.add_vline(x=-3, line=dict(color='red', width=1, dash='dash'), row=row, col=col)
        
        fig.update_xaxes(title_text="Normalized Innovation (σ)", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_confidence_evolution(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add confidence evolution plot."""
        
        conf_df = df[df['confidence'].notna()] if 'confidence' in df.columns else pd.DataFrame()
        if conf_df.empty:
            self._add_no_data_annotation(fig, row, col, "No confidence data")
            return
        
        # Color code by confidence level
        colors = ['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' 
                  for c in conf_df['confidence']]
        
        fig.add_trace(
            go.Scatter(
                x=conf_df['timestamp'],
                y=conf_df['confidence'],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=colors,
                    line=dict(width=0.5, color='white')
                ),
                line=dict(width=1, color='gray'),
                hovertemplate='<b>Confidence</b><br>Value: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add threshold lines
        fig.add_hline(y=0.8, line=dict(color='green', width=1, dash='dash'), row=row, col=col)
        fig.add_hline(y=0.5, line=dict(color='orange', width=1, dash='dash'), row=row, col=col)
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", range=[0, 1.05], row=row, col=col)
    
    def _add_component_timeline(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add component scores timeline."""
        
        components = ['safety', 'plausibility', 'consistency', 'reliability']
        colors = ['#2E7D32', '#1976D2', '#7B1FA2', '#F57C00']
        
        has_data = False
        for i, component in enumerate(components):
            if component in df.columns and not df[component].isna().all():
                comp_df = df[df[component].notna()]
                if not comp_df.empty:
                    has_data = True
                    fig.add_trace(
                        go.Scatter(
                            x=comp_df['timestamp'],
                            y=comp_df[component],
                            mode='lines+markers',
                            name=component.capitalize(),
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{component.capitalize()}</b><br>Score: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
                        ),
                        row=row, col=col
                    )
        
        if not has_data:
            self._add_no_data_annotation(fig, row, col, "No component score data")
            return
        
        # Add threshold line
        fig.add_hline(y=0.6, line=dict(color='red', width=1, dash='dash'), row=row, col=col)
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Score", range=[0, 1.05], row=row, col=col)
    
    def _add_gap_reset_table(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add table showing gap reset events."""
        
        reset_df = df[df['was_reset'] == True] if 'was_reset' in df.columns else pd.DataFrame()
        
        if reset_df.empty:
            # Show empty table with message
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Date', 'Gap (days)', 'Weight (kg)'],
                        fill_color='#E3F2FD',
                        align='left'
                    ),
                    cells=dict(
                        values=[['No gap resets'], [''], ['']],
                        fill_color='white',
                        align='left'
                    )
                ),
                row=row, col=col
            )
        else:
            # Prepare table data
            dates = reset_df['timestamp'].dt.strftime('%Y-%m-%d')
            gaps = reset_df['gap_days'].round(1) if 'gap_days' in reset_df.columns else ['N/A'] * len(reset_df)
            weights = reset_df['raw_weight'].round(1) if 'raw_weight' in reset_df.columns else ['N/A'] * len(reset_df)
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Date', 'Gap (days)', 'Weight (kg)'],
                        fill_color='#E3F2FD',
                        align='left',
                        font=dict(size=12)
                    ),
                    cells=dict(
                        values=[dates, gaps, weights],
                        fill_color='white',
                        align='left',
                        font=dict(size=11)
                    )
                ),
                row=row, col=col
            )
    
    def _add_source_impact(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add source reliability impact analysis."""
        
        if 'source' not in df.columns:
            self._add_no_data_annotation(fig, row, col, "No source data available")
            return
        
        # Calculate acceptance rate by source
        source_stats = []
        for source in df['source'].unique():
            if pd.isna(source):
                continue
            source_df = df[df['source'] == source]
            accepted = source_df[source_df['accepted'] == True] if 'accepted' in source_df.columns else pd.DataFrame()
            
            stats = {
                'source': source[:20],  # Truncate long source names
                'total': len(source_df),
                'accepted': len(accepted),
                'rate': len(accepted) / len(source_df) * 100 if len(source_df) > 0 else 0
            }
            
            # Add average quality score if available
            if 'quality_score' in source_df.columns:
                avg_quality = source_df['quality_score'].mean()
                if not pd.isna(avg_quality):
                    stats['avg_quality'] = avg_quality
            
            source_stats.append(stats)
        
        if not source_stats:
            self._add_no_data_annotation(fig, row, col, "No source statistics")
            return
        
        # Sort by total count
        source_stats.sort(key=lambda x: x['total'], reverse=True)
        source_stats = source_stats[:10]  # Top 10 sources
        
        sources = [s['source'] for s in source_stats]
        totals = [s['total'] for s in source_stats]
        rates = [s['rate'] for s in source_stats]
        
        # Create grouped bar chart
        fig.add_trace(
            go.Bar(
                x=sources,
                y=totals,
                name='Total',
                marker=dict(color='#1976D2'),
                yaxis='y',
                hovertemplate='<b>%{x}</b><br>Total: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add acceptance rate as line
        fig.add_trace(
            go.Scatter(
                x=sources,
                y=rates,
                mode='lines+markers',
                name='Accept Rate (%)',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=8),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Accept Rate: %{y:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Source", tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _prepare_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare comprehensive dataframe from results."""
        
        data = []
        for r in results:
            if not r:
                continue
            
            # Extract all available fields
            record = {
                'timestamp': pd.to_datetime(r.get('timestamp')),
                'accepted': r.get('accepted', False),
                'raw_weight': r.get('raw_weight'),
                'filtered_weight': r.get('filtered_weight'),
                'source': r.get('source', 'unknown'),
                'quality_score': r.get('quality_score'),
                'innovation': r.get('innovation'),
                'normalized_innovation': r.get('normalized_innovation'),
                'confidence': r.get('confidence', 0.5),
                'rejection_reason': r.get('reason'),
                'stage': r.get('stage'),
                'was_reset': r.get('was_reset', False),
                'gap_days': r.get('gap_days', 0)
            }
            
            # Extract quality components
            components = r.get('quality_components', {})
            for comp in ['safety', 'plausibility', 'consistency', 'reliability']:
                record[comp] = components.get(comp)
            
            # Categorize rejection
            if record['rejection_reason']:
                record['rejection_category'] = self._categorize_rejection(record['rejection_reason'])
            else:
                record['rejection_category'] = None
            
            data.append(record)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('timestamp')
        
        return df
    
    def _categorize_rejection(self, reason: str) -> str:
        """Categorize rejection reason."""
        if not reason:
            return 'Unknown'
        
        reason_lower = reason.lower()
        
        if 'quality' in reason_lower and 'score' in reason_lower:
            return 'Quality Score'
        elif 'bmi' in reason_lower:
            return 'BMI Check'
        elif 'unit' in reason_lower:
            return 'Unit Conversion'
        elif 'physiological' in reason_lower:
            return 'Physiological'
        elif 'deviation' in reason_lower:
            return 'Deviation'
        elif 'gap' in reason_lower:
            return 'Gap Validation'
        elif 'safety' in reason_lower:
            return 'Safety'
        else:
            return 'Other'
    
    def _get_rejection_color(self, category: str) -> str:
        """Get color for rejection category."""
        colors = {
            'Quality Score': '#D32F2F',
            'BMI Check': '#7B1FA2',
            'Unit Conversion': '#F57C00',
            'Physiological': '#C62828',
            'Deviation': '#FF6F00',
            'Gap Validation': '#5D4037',
            'Safety': '#B71C1C',
            'Other': '#757575',
            'Unknown': '#9E9E9E'
        }
        return colors.get(category, '#757575')
    
    def _add_no_data_annotation(self, fig: go.Figure, row: int, col: int, message: str):
        """Add no data annotation to subplot."""
        fig.add_annotation(
            text=message,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="gray"),
            row=row,
            col=col
        )

def create_diagnostic_report(results: List[Dict[str, Any]], user_id: str) -> str:
    """
    Create a text-based diagnostic report for export.
    
    Args:
        results: Processing results
        user_id: User identifier
        
    Returns:
        Formatted diagnostic report as string
    """
    
    if not results:
        return f"No data available for user {user_id}"
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(df)
    accepted = df[df['accepted'] == True] if 'accepted' in df.columns else pd.DataFrame()
    rejected = df[df['accepted'] == False] if 'accepted' in df.columns else pd.DataFrame()
    
    report = []
    report.append("=" * 80)
    report.append(f"WEIGHT PROCESSING DIAGNOSTIC REPORT")
    report.append(f"User: {user_id}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Summary Statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"Total Measurements: {total}")
    report.append(f"Accepted: {len(accepted)} ({len(accepted)/total*100:.1f}%)")
    report.append(f"Rejected: {len(rejected)} ({len(rejected)/total*100:.1f}%)")
    report.append("")
    
    # Quality Score Analysis
    if 'quality_score' in df.columns and not df['quality_score'].isna().all():
        report.append("QUALITY SCORE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mean Quality Score: {df['quality_score'].mean():.3f}")
        report.append(f"Std Dev: {df['quality_score'].std():.3f}")
        report.append(f"Min: {df['quality_score'].min():.3f}")
        report.append(f"Max: {df['quality_score'].max():.3f}")
        
        # Component breakdown
        components = ['safety', 'plausibility', 'consistency', 'reliability']
        report.append("\nComponent Scores (mean):")
        for comp in components:
            if comp in df.columns and not df[comp].isna().all():
                report.append(f"  {comp.capitalize()}: {df[comp].mean():.3f}")
        report.append("")
    
    # Rejection Analysis
    if not rejected.empty and 'reason' in rejected.columns:
        report.append("REJECTION ANALYSIS")
        report.append("-" * 40)
        
        # Count by reason
        reason_counts = rejected['reason'].value_counts()
        report.append("Top Rejection Reasons:")
        for reason, count in reason_counts.head(10).items():
            # Ensure reason fits on one line (80 char total - indent - count)
            max_reason_len = 70
            if len(reason) > max_reason_len:
                reason = reason[:max_reason_len-3] + "..."
            report.append(f"  {count:3d} - {reason}")
        report.append("")
    
    # Kalman Filter Performance
    if 'innovation' in df.columns and not df['innovation'].isna().all():
        report.append("KALMAN FILTER PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Mean Innovation: {df['innovation'].mean():.3f} kg")
        report.append(f"Std Innovation: {df['innovation'].std():.3f} kg")
        
        if 'normalized_innovation' in df.columns:
            report.append(f"Mean Normalized Innovation: {df['normalized_innovation'].mean():.2f}σ")
            report.append(f"Max Normalized Innovation: {df['normalized_innovation'].max():.2f}σ")
        
        if 'confidence' in df.columns:
            report.append(f"Mean Confidence: {df['confidence'].mean():.3f}")
        report.append("")
    
    # Gap Reset Events
    if 'was_reset' in df.columns:
        reset_df = df[df['was_reset'] == True]
        if not reset_df.empty:
            report.append("GAP RESET EVENTS")
            report.append("-" * 40)
            report.append(f"Total Resets: {len(reset_df)}")
            
            if 'gap_days' in reset_df.columns:
                report.append(f"Average Gap: {reset_df['gap_days'].mean():.1f} days")
                report.append(f"Max Gap: {reset_df['gap_days'].max():.1f} days")
            report.append("")
    
    # Source Analysis
    if 'source' in df.columns:
        report.append("SOURCE ANALYSIS")
        report.append("-" * 40)
        
        source_counts = df['source'].value_counts()
        report.append("Measurements by Source:")
        for source, count in source_counts.head(10).items():
            if 'accepted' in df.columns:
                source_df = df[df['source'] == source]
                source_accepted = source_df[source_df['accepted'] == True]
                rate = len(source_accepted) / len(source_df) * 100
                report.append(f"  {source[:30]:30s}: {count:4d} ({rate:.1f}% accepted)")
            else:
                report.append(f"  {source[:30]:30s}: {count:4d}")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    
    return "\n".join(report)
