#!/usr/bin/env python3
"""
Parallel execution utilities for running analysis steps concurrently.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional


class ParallelExecutor:
    """Execute analysis steps in parallel with error handling."""

    def __init__(self, max_workers: int = 3):
        """Initialize with maximum number of worker threads."""
        self.max_workers = max_workers
        self.errors: List[Tuple[str, Exception]] = []

    def execute_tasks(self, tasks: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: Dictionary mapping task names to callable functions

        Returns:
            Dictionary of task results
        """
        results = {}
        self.errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(func): name
                for name, func in tasks.items()
            }

            # Wait for completion
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result_type, result_data = future.result()

                    if result_type.endswith('_error'):
                        self.errors.append((task_name, result_data))
                    else:
                        results[result_type] = result_data

                except Exception as e:
                    logging.error(f"Task {task_name} failed with unexpected error: {e}")
                    self.errors.append((task_name, e))

        return results

    def report_errors(self):
        """Log any errors that occurred during parallel execution."""
        if self.errors:
            logging.warning(f"\n{len(self.errors)} parallel tasks failed:")
            for task_name, error in self.errors:
                logging.warning(f"  - {task_name}: {error}")
            logging.warning("Continuing with report generation...")

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


def create_daily_analysis_task(user_start_dates: Dict[str, str], output_path: Path) -> Callable:
    """Create task for daily analysis generation."""
    def run_daily_analysis():
        try:
            import generate_daily_analysis
            logging.info("Step 1b: Starting daily detail report generation...")
            result = generate_daily_analysis.main(user_start_dates, output_path)
            logging.info(f"Step 1b: Complete - {result.get('total_records', 0):,} records generated")
            return ('daily', result)
        except Exception as e:
            logging.error(f"Step 1b failed: {e}")
            return ('daily_error', e)
    return run_daily_analysis


def create_visualizations_task(analysis_file: Path, output_path: Path) -> Callable:
    """Create task for visualization generation."""
    def run_visualizations():
        try:
            import generate_visualizations
            logging.info("Step 2: Starting visualization generation...")
            generate_visualizations.main(analysis_file, output_path)
            logging.info("Step 2: Complete - all visualizations generated")
            return ('viz', True)
        except Exception as e:
            logging.error(f"Step 2 failed: {e}")
            return ('viz_error', e)
    return run_visualizations


def create_statistical_report_task(output_path: Path) -> Callable:
    """Create task for statistical report generation."""
    def run_statistical_report():
        try:
            import generate_statistical_report
            logging.info("Step 3: Starting statistical evidence analysis...")
            generate_statistical_report.generate_report(output_path)
            logging.info("Step 3: Complete - statistical report generated")
            return ('stat', True)
        except Exception as e:
            logging.error(f"Step 3 failed: {e}")
            return ('stat_error', e)
    return run_statistical_report


def run_parallel_analysis(
    user_start_dates: Dict[str, str],
    output_path: Path,
    visualizations_path: Path,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Run parallel analysis steps (daily analysis, visualizations, statistical report).

    Args:
        user_start_dates: Dictionary of user IDs to start dates
        output_path: Main output directory path
        visualizations_path: Visualizations output directory path
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary containing results from successful tasks
    """
    logging.info("\n" + "="*50)
    logging.info("RUNNING PARALLEL ANALYSIS STEPS (1b, 2, 3)")
    logging.info("="*50)

    executor = ParallelExecutor(max_workers)

    # Create tasks
    tasks = {
        'daily_analysis': create_daily_analysis_task(user_start_dates, output_path),
        'visualizations': create_visualizations_task(
            output_path / "90_day_analysis.csv",
            visualizations_path
        ),
        'statistical_report': create_statistical_report_task(output_path)
    }

    # Execute in parallel
    results = executor.execute_tasks(tasks)

    # Report any errors
    executor.report_errors()

    logging.info("\nParallel analysis steps complete")

    return results