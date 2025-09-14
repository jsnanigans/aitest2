"""
Simplified diagnostic dashboard that works around plotly subplot limitations.
Uses a simpler layout to avoid issues with mixed subplot types.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class SimpleDiagnosticDashboard:
    """Simplified diagnostic dashboard that avoids subplot type conflicts."""
    
    def create_dashboard(
        self,
        results: List[Dict[str, Any]],
        user_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create simplified diagnostic dashboard."""
        
        if not results:
            return self._create_empty_dashboard(user_id)
        
        # Prepare data
        df = self._prepare_dataframe(results)
        
        # Create figure with simpler layout - all scatter/bar types
        fig = make_subplots(
            rows=3, cols=3,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=(
                "Weight Processing Timeline",
                "Quality Score Components", 
                "Rejection Categories",
                "Kalman Innovation",
                "Innovation Distribution",
                "Confidence Score",
                "Source Acceptance Rates",
                "Quality Score Timeline",
                "Processing Summary"
            ),
            specs=[
                [{"colspan": 3}, None, None],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.12
        )
        
        # Add plots
        self._add_main_timeline(fig, df, row=1, col=1)
        self._add_kalman_innovation(fig, df, row=2, col=1)
        self._add_innovation_histogram(fig, df, row=2, col=2)
        self._add_confidence_plot(fig, df, row=2, col=3)
        self._add_source_analysis(fig, df, row=3, col=1)
        self._add_quality_timeline(fig, df, row=3, col=2)
        self._add_summary_metrics(fig, df, row=3, col=3)
        
        # Update layout with improved styling
        fig.update_layout(
            title=dict(
                text=f"Weight Processing Diagnostic Dashboard - User {user_id[:8]}",
                font=dict(size=18, color='#2c3e50', family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            height=1200,
            showlegend=True,
            template='plotly_white',
            hovermode='closest',  # Better hover behavior
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(
                family='Arial, sans-serif',
                size=11,
                color='#2c3e50'
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update all subplot backgrounds
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        
        return fig
    
    def _create_empty_dashboard(self, user_id: str) -> go.Figure:
        """Create empty dashboard."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title=f"Diagnostic Dashboard - User {user_id[:8]}",
            height=600
        )
        return fig
    
    def _add_main_timeline(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add main timeline with decisions."""
        
        # Accepted measurements
        accepted_df = df[df['accepted'] == True]
        if not accepted_df.empty:
            # Kalman filtered line
            if 'filtered_weight' in accepted_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_df['timestamp'],
                        y=accepted_df['filtered_weight'],
                        mode='lines',
                        name='Kalman Filtered',
                        line=dict(color='#1565C0', width=2)
                    ),
                    row=row, col=col
                )
            
            # Raw measurements with improved quality color scheme
            colors = accepted_df['quality_score'].fillna(0.5) if 'quality_score' in accepted_df.columns else [0.5] * len(accepted_df)
            
            # Create detailed hover text with all available information
            hover_texts = []
            for idx, row_data in accepted_df.iterrows():
                hover_parts = [
                    f"<b>Weight: {row_data['raw_weight']:.2f} kg</b>",
                    f"Date: {row_data['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    f"Source: {row_data.get('source', 'unknown')[:30]}",
                    f"Quality Score: {row_data.get('quality_score', 0):.3f}"
                ]
                
                # Add quality components if available
                if 'safety' in row_data and pd.notna(row_data['safety']):
                    hover_parts.append(f"<b>Quality Components:</b>")
                    hover_parts.append(f"  Safety: {row_data.get('safety', 0):.2f}")
                    hover_parts.append(f"  Plausibility: {row_data.get('plausibility', 0):.2f}")
                    hover_parts.append(f"  Consistency: {row_data.get('consistency', 0):.2f}")
                    hover_parts.append(f"  Reliability: {row_data.get('reliability', 0):.2f}")
                
                # Add Kalman metrics if available
                if 'innovation' in row_data and pd.notna(row_data['innovation']):
                    hover_parts.append(f"<b>Kalman Metrics:</b>")
                    hover_parts.append(f"  Innovation: {row_data['innovation']:.3f} kg")
                    hover_parts.append(f"  Norm. Innovation: {row_data.get('normalized_innovation', 0):.2f}σ")
                    hover_parts.append(f"  Confidence: {row_data.get('confidence', 0):.2f}")
                
                hover_texts.append("<br>".join(hover_parts))
            
            fig.add_trace(
                go.Scatter(
                    x=accepted_df['timestamp'],
                    y=accepted_df['raw_weight'],
                    mode='markers',
                    name='Accepted',
                    marker=dict(
                        size=12,
                        color=colors,
                        colorscale=[
                            [0.0, '#d73027'],  # Red for very low quality
                            [0.3, '#fc8d59'],  # Orange for low quality
                            [0.5, '#fee08b'],  # Yellow for medium quality
                            [0.7, '#d9ef8b'],  # Light green for good quality
                            [0.85, '#91cf60'], # Green for high quality
                            [1.0, '#1a9850']   # Dark green for excellent quality
                        ],
                        cmin=0,
                        cmax=1,
                        colorbar=dict(
                            title='Quality<br>Score',
                            thickness=15,
                            len=0.4,
                            y=0.85,
                            yanchor='top',
                            tickmode='array',
                            tickvals=[0, 0.3, 0.6, 0.8, 1.0],
                            ticktext=['0.0', '0.3', '0.6', '0.8', '1.0']
                        ),
                        line=dict(color='white', width=1.5),
                        opacity=0.9
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Rejected measurements with detailed information
        rejected_df = df[df['accepted'] == False]
        if not rejected_df.empty:
            # Create detailed hover text for rejections
            rejection_hover = []
            for idx, row_data in rejected_df.iterrows():
                hover_parts = [
                    f"<b>REJECTED - Weight: {row_data['raw_weight']:.2f} kg</b>",
                    f"Date: {row_data['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    f"Source: {row_data.get('source', 'unknown')[:30]}",
                    f"<b>Rejection Reason:</b>",
                    f"{row_data.get('rejection_reason', 'Unknown')}"
                ]
                
                # Add quality score if available
                if 'quality_score' in row_data and pd.notna(row_data['quality_score']):
                    hover_parts.append(f"Quality Score: {row_data['quality_score']:.3f}")
                
                rejection_hover.append("<br>".join(hover_parts))
            
            fig.add_trace(
                go.Scatter(
                    x=rejected_df['timestamp'],
                    y=rejected_df['raw_weight'],
                    mode='markers',
                    name='Rejected',
                    marker=dict(
                        size=10,
                        symbol='x',
                        color='#e41a1c',  # Bright red
                        line=dict(width=2, color='darkred'),
                        opacity=0.8
                    ),
                    text=rejection_hover,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Gap resets as vertical markers with improved visibility
        reset_df = df[df['was_reset'] == True] if 'was_reset' in df.columns else pd.DataFrame()
        if not reset_df.empty:
            y_min = df['raw_weight'].min()
            y_max = df['raw_weight'].max()
            y_range = y_max - y_min
            
            for _, reset in reset_df.iterrows():
                gap_days = reset.get('gap_days', 0)
                # Add as a separate trace for the vertical line effect
                fig.add_trace(
                    go.Scatter(
                        x=[reset['timestamp'], reset['timestamp']],
                        y=[y_min - y_range * 0.05, y_max + y_range * 0.05],
                        mode='lines+text',
                        line=dict(color='#7f7f7f', width=2, dash='dot'),
                        text=['', f'Reset: {gap_days:.0f}d gap'],
                        textposition='top center',
                        textfont=dict(size=10, color='#7f7f7f'),
                        showlegend=False,
                        hovertemplate=f'<b>State Reset</b><br>Gap: {gap_days:.0f} days<br>Date: {reset["timestamp"].strftime("%Y-%m-%d")}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            # Add one legend entry for all resets
            if len(reset_df) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='lines',
                        line=dict(color='#7f7f7f', width=2, dash='dot'),
                        name=f'State Resets ({len(reset_df)})',
                        showlegend=True
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Weight (kg)", row=row, col=col)
    
    def _add_kalman_innovation(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add Kalman innovation plot."""
        
        inn_df = df[df['innovation'].notna()] if 'innovation' in df.columns else pd.DataFrame()
        if inn_df.empty:
            return
        
        colors = ['#1a9850' if abs(i) < 1 else '#fdae61' if abs(i) < 2 else '#d73027' 
                  for i in inn_df['innovation']]
        
        fig.add_trace(
            go.Scatter(
                x=inn_df['timestamp'],
                y=inn_df['innovation'],
                mode='markers+lines',
                marker=dict(
                    size=8, 
                    color=colors,
                    line=dict(width=1, color='white')
                ),
                line=dict(width=1.5, color='#95a5a6'),
                name='Innovation',
                hovertemplate='<b>Innovation</b><br>Value: %{y:.3f} kg<br>Date: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add zero line as trace
        fig.add_trace(
            go.Scatter(
                x=[inn_df['timestamp'].min(), inn_df['timestamp'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Innovation (kg)", row=row, col=col)
    
    def _add_innovation_histogram(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add innovation distribution."""
        
        norm_inn = df['normalized_innovation'].dropna() if 'normalized_innovation' in df.columns else pd.Series()
        if norm_inn.empty:
            return
        
        fig.add_trace(
            go.Bar(
                x=np.histogram(norm_inn, bins=20)[1][:-1],
                y=np.histogram(norm_inn, bins=20)[0],
                name='Distribution',
                marker=dict(color='#1976D2')
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Normalized Innovation (σ)", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_confidence_plot(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add confidence evolution."""
        
        conf_df = df[df['confidence'].notna()] if 'confidence' in df.columns else pd.DataFrame()
        if conf_df.empty:
            return
        
        fig.add_trace(
            go.Scatter(
                x=conf_df['timestamp'],
                y=conf_df['confidence'],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(width=2, color='#7B1FA2'),
                name='Confidence',
                hovertemplate='Confidence: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add threshold lines as traces
        x_range = [conf_df['timestamp'].min(), conf_df['timestamp'].max()]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[0.8, 0.8],
                mode='lines',
                line=dict(color='green', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Confidence", range=[0, 1.05], row=row, col=col)
    
    def _add_source_analysis(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add source acceptance rate analysis."""
        
        if 'source' not in df.columns:
            return
        
        source_stats = []
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            accepted = source_df[source_df['accepted'] == True] if 'accepted' in source_df.columns else pd.DataFrame()
            
            source_stats.append({
                'source': source[:15],  # Truncate
                'total': len(source_df),
                'rate': len(accepted) / len(source_df) * 100 if len(source_df) > 0 else 0
            })
        
        source_stats.sort(key=lambda x: x['total'], reverse=True)
        source_stats = source_stats[:8]  # Top 8
        
        sources = [s['source'] for s in source_stats]
        rates = [s['rate'] for s in source_stats]
        
        fig.add_trace(
            go.Bar(
                x=sources,
                y=rates,
                name='Accept Rate',
                marker=dict(color='#4CAF50'),
                text=[f"{r:.0f}%" for r in rates],
                textposition='outside',
                hovertemplate='%{x}<br>Accept: %{y:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Source", tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text="Accept Rate (%)", range=[0, 105], row=row, col=col)
    
    def _add_quality_timeline(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add quality score timeline."""
        
        quality_df = df[df['quality_score'].notna()] if 'quality_score' in df.columns else pd.DataFrame()
        if quality_df.empty:
            return
        
        fig.add_trace(
            go.Scatter(
                x=quality_df['timestamp'],
                y=quality_df['quality_score'],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(width=2, color='#FF6F00'),
                name='Quality Score',
                hovertemplate='Quality: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add threshold line
        x_range = [quality_df['timestamp'].min(), quality_df['timestamp'].max()]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[0.6, 0.6],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Threshold',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Quality Score", range=[0, 1.05], row=row, col=col)
    
    def _add_summary_metrics(self, fig: go.Figure, df: pd.DataFrame, row: int, col: int):
        """Add summary metrics as a simple scatter plot with text."""
        
        # Calculate metrics
        total = len(df)
        accepted = len(df[df['accepted'] == True]) if 'accepted' in df.columns else 0
        rejected = total - accepted
        accept_rate = accepted / total * 100 if total > 0 else 0
        
        avg_quality = df['quality_score'].mean() if 'quality_score' in df.columns else 0
        
        # Create text display
        metrics = [
            f"Total: {total}",
            f"Accepted: {accepted} ({accept_rate:.1f}%)",
            f"Rejected: {rejected}",
            f"Avg Quality: {avg_quality:.2f}"
        ]
        
        # Display as scatter with text
        fig.add_trace(
            go.Scatter(
                x=[0.5] * len(metrics),
                y=list(range(len(metrics), 0, -1)),
                mode='text',
                text=metrics,
                textfont=dict(size=14),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    
    def _prepare_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare dataframe from results."""
        
        data = []
        for r in results:
            if not r:
                continue
            
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
                'was_reset': r.get('was_reset', False),
                'gap_days': r.get('gap_days', 0)
            }
            
            # Extract quality components
            components = r.get('quality_components', {})
            for comp in ['safety', 'plausibility', 'consistency', 'reliability']:
                record[comp] = components.get(comp)
            
            data.append(record)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('timestamp')
        
        return df