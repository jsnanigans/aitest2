#!/usr/bin/env python3
"""
Shared Data Cache Module
Provides singleton cache for CSV data to avoid redundant loading across modules
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


class DataCache:
    """Singleton cache for shared data files."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: Dict[str, pd.DataFrame] = {}
        self._file_locks: Dict[str, threading.Lock] = {}
        self._initialized = True

    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file."""
        if file_path not in self._file_locks:
            with self._lock:
                if file_path not in self._file_locks:
                    self._file_locks[file_path] = threading.Lock()
        return self._file_locks[file_path]

    def get_dataframe(
        self,
        file_path: Path,
        usecols: Optional[list] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Get a dataframe from cache or load it if not cached.

        Args:
            file_path: Path to the CSV file
            usecols: Columns to load (if None, loads all)
            force_reload: Force reload from disk

        Returns:
            DataFrame (may be a view if usecols is specified)
        """
        # Create cache key based on file path and columns
        cache_key = str(file_path)
        if usecols:
            cache_key += f"_cols_{','.join(sorted(usecols))}"

        # Get file-specific lock
        file_lock = self._get_file_lock(str(file_path))

        with file_lock:
            # Check if we need to load
            if force_reload or cache_key not in self._cache:
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                logging.info(f"Loading {file_path.name} into cache...")
                df = pd.read_csv(file_path, usecols=usecols)

                # Convert datetime columns if present
                datetime_cols = ['effectiveDateTime', 'start_date']
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])

                self._cache[cache_key] = df
                logging.info(f"Cached {len(df):,} rows from {file_path.name}")

            # Return a copy to prevent modifications affecting cache
            return self._cache[cache_key].copy()

    def get_raw_and_filtered(
        self,
        raw_path: Path,
        filtered_path: Path,
        usecols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get both raw and filtered dataframes.

        Args:
            raw_path: Path to raw CSV
            filtered_path: Path to filtered CSV
            usecols: Columns to load

        Returns:
            Tuple of (raw_df, filtered_df)
        """
        df_raw = self.get_dataframe(raw_path, usecols)
        df_filtered = self.get_dataframe(filtered_path, usecols)
        return df_raw, df_filtered

    def preload_all(
        self,
        raw_path: Path,
        filtered_path: Path,
        user_employers_path: Optional[Path] = None
    ):
        """
        Preload all commonly used files into cache.

        Args:
            raw_path: Path to raw CSV
            filtered_path: Path to filtered CSV
            user_employers_path: Optional path to user employers CSV
        """
        logging.info("Preloading data files into cache...")

        # Load weight data files with commonly needed columns
        weight_cols = ['user_id', 'effectiveDateTime', 'weight']
        self.get_dataframe(raw_path, weight_cols)
        self.get_dataframe(filtered_path, weight_cols)

        # Load user employers if provided
        if user_employers_path and user_employers_path.exists():
            self.get_dataframe(user_employers_path)

        logging.info("Data preloading complete")

    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            logging.info("Cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached dataframes."""
        return len(self._cache)

    def get_memory_usage(self) -> float:
        """Get total memory usage in MB."""
        total_bytes = sum(
            df.memory_usage(deep=True).sum()
            for df in self._cache.values()
        )
        return total_bytes / (1024 * 1024)


# Global singleton instance
data_cache = DataCache()