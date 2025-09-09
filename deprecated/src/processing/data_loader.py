#!/usr/bin/env python3

import csv
from typing import Iterator, Dict, Any, Set, Optional
from src.core import get_logger

logger = get_logger(__name__)


from datetime import datetime

class StreamingCSVReader:
    def __init__(self, input_file: str, encoding: str = "utf-8", max_date: str = None):
        self.input_file = input_file
        self.encoding = encoding
        self.total_rows = 0
        self.current_user_id = None
        self.max_date = None
        self.skipped_future_dates = 0
        
        # Parse max_date if provided
        if max_date:
            try:
                self.max_date = datetime.strptime(max_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                logger.warning(f"Invalid max_date format: {max_date}, ignoring date filter")
                self.max_date = None
        
    def read_rows(self) -> Iterator[Dict[str, Any]]:
        with open(self.input_file, encoding=self.encoding) as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                self.total_rows += 1
                
                # Skip rows with dates beyond max_date if configured
                if self.max_date:
                    date_field = row.get('effectivDateTime') or row.get('date')
                    if date_field:
                        try:
                            row_date = datetime.strptime(date_field, "%Y-%m-%d %H:%M:%S")
                            if row_date > self.max_date:
                                self.skipped_future_dates += 1
                                logger.debug(f"Skipping future date: {date_field} > {self.max_date.strftime('%Y-%m-%d')}")
                                continue  # Skip this row entirely
                        except (ValueError, TypeError):
                            # If we can't parse the date, let it through for downstream handling
                            pass
                
                yield row
                
    def get_total_rows(self) -> int:
        return self.total_rows
    
    def get_skipped_stats(self) -> Dict[str, int]:
        """Return statistics about skipped rows"""
        return {
            'future_dates': self.skipped_future_dates,
            'total_skipped': self.skipped_future_dates
        }


class UserFilteredReader:
    def __init__(
        self, 
        reader: StreamingCSVReader,
        specific_user_ids: Optional[Set[str]] = None,
        skip_first_users: int = 0,
        max_users: int = 0
    ):
        self.reader = reader
        self.specific_user_ids = specific_user_ids
        self.skip_first_users = skip_first_users
        self.max_users = max_users
        self.users_seen = set()
        self.users_skipped = 0
        self.users_processed = 0
        
    def should_process_user(self, user_id: str) -> bool:
        if self.specific_user_ids and user_id not in self.specific_user_ids:
            return False
            
        if user_id not in self.users_seen:
            self.users_seen.add(user_id)
            if not self.specific_user_ids and len(self.users_seen) <= self.skip_first_users:
                self.users_skipped += 1
                logger.info(f"Skipping user {user_id} ({self.users_skipped}/{self.skip_first_users})")
                return False
                
        if self.max_users and self.users_processed >= self.max_users:
            return False
            
        return True
        
    def mark_user_processed(self):
        self.users_processed += 1
        
    def filter_rows(self) -> Iterator[Dict[str, Any]]:
        current_user_id = None
        user_being_processed = False
        
        for row in self.reader.read_rows():
            user_id = row["user_id"]
            
            if user_id != current_user_id:
                current_user_id = user_id
                user_being_processed = self.should_process_user(user_id)
                
            if user_being_processed:
                yield row
                
            if self.max_users and self.users_processed >= self.max_users:
                break