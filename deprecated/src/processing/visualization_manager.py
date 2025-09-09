#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, Any
from src.core import get_logger

logger = get_logger(__name__)

try:
    from src.visualization.dashboard import HAS_MATPLOTLIB, create_unified_visualization
except ImportError:
    HAS_MATPLOTLIB = False
    create_unified_visualization = None
    logger.warning("Visualization module not available")


class VisualizationManager:
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.enable_visualization = config.get("enable_visualization", True) and HAS_MATPLOTLIB
        self.max_visualizations = config.get("max_visualizations", 20)
        self.viz_dir = None
        self.visualizations_created = 0
        
        if self.enable_visualization:
            self._setup_viz_directory()
            
    def _setup_viz_directory(self):
        if self.config.get("use_test_output_names", False):
            self.viz_dir = self.output_dir / "dashboards_test"
        else:
            timestamp = "yes"  # Will be passed in from main
            pattern = self.config.get("dashboard_dir_pattern", "dashboards_{timestamp}")
            self.viz_dir = self.output_dir / pattern.format(timestamp=timestamp)
        self.viz_dir.mkdir(exist_ok=True)
        
    def create_visualization(self, user_id: str, user_stats: Dict[str, Any]) -> bool:
        if not self.enable_visualization or not self.viz_dir:
            return False
            
        if self.visualizations_created >= self.max_visualizations:
            return False
            
        if create_unified_visualization is None:
            return False
            
        try:
            create_unified_visualization(user_id, user_stats, self.viz_dir, self.config)
            self.visualizations_created += 1
            return True
        except Exception as e:
            logger.warning(f"Visualization failed for {user_id}: {e}")
            return False
            
    def get_viz_directory(self) -> Path:
        return self.viz_dir