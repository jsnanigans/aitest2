import os
import sys
from typing import Dict, List, Any, Optional
import toml
from pathlib import Path

try:
    from .viz_logger import get_logger, set_verbosity
    logger = get_logger()
except ImportError:
    # Fallback if logger not available
    class SimpleLogger:
        def warning(self, msg): print(f"Warning: {msg}")
        def error(self, msg): print(f"Error: {msg}")
    logger = SimpleLogger()
    def set_verbosity(level): pass

def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        return toml.load(config_path)
    return {}

def should_use_interactive(config: Optional[Dict[str, Any]] = None, 
                          output_format: Optional[str] = None) -> bool:
    if config is None:
        config = load_config()
    
    viz_config = config.get("visualization", {})
    mode = viz_config.get("mode", "auto")
    
    if mode == "interactive":
        return True
    elif mode == "static":
        return False
    else:  # auto mode
        if output_format:
            return output_format.lower() in ["html", "interactive", "plotly"]
        
        if "JUPYTER_RUNTIME_DIR" in os.environ:
            return True
        
        if hasattr(sys, 'ps1'):
            return True
        
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        
        return False

def create_dashboard(results: List[Dict[str, Any]], 
                    user_id: str,
                    output_dir: str = "output",
                    config: Optional[Dict[str, Any]] = None,
                    output_format: Optional[str] = None) -> str:
    if config is None:
        config = load_config()
    
    use_interactive = should_use_interactive(config, output_format)
    
    # Check if we should use diagnostic dashboard (new default)
    use_diagnostic = config.get("visualization", {}).get("use_diagnostic", True)
    use_enhanced = config.get("visualization", {}).get("use_enhanced", True)
    
    if use_interactive:
        try:
            if use_diagnostic:
                # Use simplified diagnostic dashboard as primary visualization
                try:
                    from .viz_diagnostic_simple import SimpleDiagnosticDashboard
                    dashboard = SimpleDiagnosticDashboard()
                    fig = dashboard.create_dashboard(results, user_id, config)
                    
                    # Save to HTML
                    output_path = Path(output_dir)
                    output_path.mkdir(exist_ok=True, parents=True)
                    dashboard_file = output_path / f"{user_id}.html"
                    fig.write_html(str(dashboard_file))
                    
                    # Also create diagnostic report
                    try:
                        from .viz_diagnostic import create_diagnostic_report
                        report = create_diagnostic_report(results, user_id)
                        report_file = output_path / f"{user_id}_report.txt"
                        with open(report_file, "w") as f:
                            f.write(report)
                    except:
                        pass  # Report is optional
                    
                    return str(dashboard_file)
                except (ImportError, Exception) as e:
                    logger.warning(f"Diagnostic dashboard failed: {e}, falling back to enhanced")
                    # Fall back to enhanced dashboard
                    if use_enhanced:
                        try:
                            from .viz_plotly_enhanced import create_enhanced_dashboard
                            return create_enhanced_dashboard(results, user_id, output_dir, config)
                        except:
                            pass
                    # Fall back to standard interactive
                    from .viz_plotly import create_interactive_dashboard
                    return create_interactive_dashboard(results, user_id, output_dir, config)
            elif use_enhanced:
                # Try to use enhanced dashboard with Kalman insights
                try:
                    from .viz_plotly_enhanced import create_enhanced_dashboard
                    return create_enhanced_dashboard(results, user_id, output_dir, config)
                except (ImportError, Exception) as e:
                    # Fall back to standard interactive if enhanced fails silently
                    from .viz_plotly import create_interactive_dashboard
                    return create_interactive_dashboard(results, user_id, output_dir, config)
            else:
                from .viz_plotly import create_interactive_dashboard
                return create_interactive_dashboard(results, user_id, output_dir, config)
        except ImportError:
            logger.warning("Plotly not available, falling back to static visualization")
            from .visualization import create_dashboard as create_static_dashboard
            return create_static_dashboard(results, user_id, output_dir, config)
    else:
        from .visualization import create_dashboard as create_static_dashboard
        return create_static_dashboard(results, user_id, output_dir, config)

def export_dashboard(dashboard_path: str, 
                    export_format: str,
                    output_path: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None) -> str:
    if config is None:
        config = load_config()
    
    export_config = config.get("visualization", {}).get("export", {})
    
    if dashboard_path.endswith(".html"):
        from .viz_export import export_plotly_dashboard
        return export_plotly_dashboard(dashboard_path, export_format, output_path, export_config)
    else:
        if export_format.lower() in ["html", "interactive"]:
            print(f"Cannot export static dashboard to {export_format} format")
            return dashboard_path
        return dashboard_path