import os
from pathlib import Path
from typing import Dict, Any, Optional

def export_plotly_dashboard(dashboard_path: str, 
                           export_format: str,
                           output_path: Optional[str] = None,
                           export_config: Dict[str, Any] = {}) -> str:
    
    dashboard_path = Path(dashboard_path)
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")
    
    if output_path is None:
        output_path = dashboard_path.with_suffix(f".{export_format.lower()}")
    else:
        output_path = Path(output_path)
    
    export_format = export_format.lower()
    
    if export_format in ["png", "jpg", "jpeg", "svg", "pdf"]:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            import json
            import re
            plot_json_match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])', html_content, re.DOTALL)
            if plot_json_match:
                plot_data = json.loads(plot_json_match.group(1))
                
                fig = go.Figure(data=plot_data)
                
                width = export_config.get("png_width", 1920)
                height = export_config.get("png_height", 1080)
                scale = export_config.get("png_scale", 2)
                
                if export_format == "pdf":
                    fig.write_image(str(output_path), format="pdf", width=width, height=height)
                elif export_format in ["jpg", "jpeg"]:
                    fig.write_image(str(output_path), format="jpeg", width=width, height=height, scale=scale)
                elif export_format == "svg":
                    fig.write_image(str(output_path), format="svg", width=width, height=height)
                else:  # png
                    fig.write_image(str(output_path), format="png", width=width, height=height, scale=scale)
                
                print(f"Dashboard exported to {output_path}")
                return str(output_path)
            else:
                print("Could not extract plot data from HTML file")
                return str(dashboard_path)
                
        except ImportError:
            print(f"Kaleido is required for {export_format} export. Install with: pip install kaleido")
            return str(dashboard_path)
        except Exception as e:
            print(f"Error exporting to {export_format}: {e}")
            return str(dashboard_path)
    
    elif export_format == "csv":
        try:
            import pandas as pd
            import json
            import re
            
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            plot_json_match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])', html_content, re.DOTALL)
            if plot_json_match:
                plot_data = json.loads(plot_json_match.group(1))
                
                data_frames = []
                for trace in plot_data:
                    if 'x' in trace and 'y' in trace:
                        df = pd.DataFrame({
                            'x': trace['x'],
                            'y': trace['y'],
                            'name': trace.get('name', 'data')
                        })
                        data_frames.append(df)
                
                if data_frames:
                    combined_df = pd.concat(data_frames, ignore_index=True)
                    combined_df.to_csv(output_path, index=False)
                    print(f"Data exported to {output_path}")
                    return str(output_path)
            
            print("Could not extract data from HTML file")
            return str(dashboard_path)
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return str(dashboard_path)
    
    elif export_format == "json":
        try:
            import json
            import re
            
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            plot_json_match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])', html_content, re.DOTALL)
            if plot_json_match:
                plot_data = json.loads(plot_json_match.group(1))
                
                with open(output_path, 'w') as f:
                    json.dump(plot_data, f, indent=2)
                
                print(f"Data exported to {output_path}")
                return str(output_path)
            
            print("Could not extract data from HTML file")
            return str(dashboard_path)
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return str(dashboard_path)
    
    else:
        print(f"Unsupported export format: {export_format}")
        return str(dashboard_path)