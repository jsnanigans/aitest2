"""
Progress indication utilities for CLI output.
Provides progress bars, spinners, and real-time statistics.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import shutil


class ProgressIndicator:
    """Enhanced progress indication for CLI processing."""
    
    def __init__(self, total_items: Optional[int] = None, show_eta: bool = True):
        self.total_items = total_items
        self.current_item = 0
        self.start_time = time.time()
        self.show_eta = show_eta
        self.last_update_time = 0
        self.update_interval = 0.1  # Update every 100ms minimum
        
        # Terminal width
        self.terminal_width = shutil.get_terminal_size(fallback=(80, 20)).columns
        
        # Colors (ANSI escape codes)
        self.COLORS = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'RESET': '\033[0m',
        }
        
        # Spinner frames
        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
        
        # Statistics
        self.stats = {
            'users_processed': 0,
            'rows_processed': 0,
            'accepted': 0,
            'rejected': 0,
            'current_user': None,
            'errors': 0
        }
        
    def start(self, message: str = "Processing"):
        """Start progress indication."""
        self.start_time = time.time()
        self.current_item = 0
        print(f"\n{self.COLORS['BOLD']}{message}...{self.COLORS['RESET']}")
        
    def update(self, current: int, extra_info: Optional[Dict[str, Any]] = None):
        """Update progress with current position."""
        self.current_item = current
        
        # Update stats if provided
        if extra_info:
            self.stats.update(extra_info)
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time
        
        # Calculate progress
        elapsed = current_time - self.start_time
        
        if self.total_items:
            # Progress bar mode
            self._draw_progress_bar(elapsed)
        else:
            # Spinner mode
            self._draw_spinner(elapsed)
            
    def _draw_progress_bar(self, elapsed: float):
        """Draw a progress bar with statistics."""
        if not self.total_items:
            return
            
        # Calculate percentage
        percentage = (self.current_item / self.total_items) * 100
        
        # Calculate rate and ETA
        rate = self.current_item / elapsed if elapsed > 0 else 0
        remaining = self.total_items - self.current_item
        eta_seconds = remaining / rate if rate > 0 else 0
        
        # Format ETA
        if self.show_eta and eta_seconds > 0:
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "--:--"
            
        # Progress bar width (leave space for text)
        bar_width = min(30, self.terminal_width - 60)
        filled = int(bar_width * (self.current_item / self.total_items))
        
        # Choose color based on progress
        if percentage < 33:
            color = self.COLORS['RED']
        elif percentage < 66:
            color = self.COLORS['YELLOW']
        else:
            color = self.COLORS['GREEN']
            
        # Build progress bar
        bar = f"{color}{'█' * filled}{'░' * (bar_width - filled)}{self.COLORS['RESET']}"
        
        # Build status line - adapt based on whether we're tracking users or rows
        if self.stats.get('users_processed') is not None and self.total_items <= 1000:
            # Likely tracking users (max_users is typically < 1000)
            status = f"\r{bar} {percentage:5.1f}% | {self.COLORS['CYAN']}Users: {self.current_item:,}/{self.total_items:,}{self.COLORS['RESET']}"
            status += f" | Rows: {self.stats.get('rows_processed', 0):,}"
        else:
            # Tracking rows
            status = f"\r{bar} {percentage:5.1f}% | Rows: {self.current_item:,}/{self.total_items:,}"
            # Add user count prominently
            if self.stats.get('users_processed'):
                status += f" | {self.COLORS['CYAN']}Users: {self.stats['users_processed']:,}{self.COLORS['RESET']}"
            
        status += f" | ETA: {eta_str}"
        
        # Add current user if available
        if self.stats.get('current_user'):
            status += f" | Current: {self.stats['current_user'][:12]:12}"
            
        # Clear line and print
        sys.stdout.write('\r' + ' ' * self.terminal_width)
        sys.stdout.write(status)
        sys.stdout.flush()
        
    def _draw_spinner(self, elapsed: float):
        """Draw a spinner with statistics for unknown total."""
        # Get spinner frame
        spinner = self.spinner_frames[self.spinner_index]
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        
        # Format elapsed time
        elapsed_str = self._format_time(elapsed)
        
        # Build status with color
        status = f"\r{self.COLORS['CYAN']}{spinner}{self.COLORS['RESET']} "
        status += f"Rows: {self.stats.get('rows_processed', 0):,} | "
        status += f"{self.COLORS['CYAN']}Users: {self.stats.get('users_processed', 0):,}{self.COLORS['RESET']} | "
        
        # Add acceptance rate if available
        accepted = self.stats.get('accepted', 0)
        rejected = self.stats.get('rejected', 0)
        total = accepted + rejected
        if total > 0:
            acc_rate = (accepted / total) * 100
            if acc_rate >= 90:
                acc_color = self.COLORS['GREEN']
            elif acc_rate >= 70:
                acc_color = self.COLORS['YELLOW']
            else:
                acc_color = self.COLORS['RED']
            status += f"Accept: {acc_color}{acc_rate:.0f}%{self.COLORS['RESET']} | "
        
        status += f"Time: {elapsed_str}"
        
        if self.stats.get('current_user'):
            status += f" | Current: {self.stats['current_user'][:20]}"
            
        # Clear line and print
        sys.stdout.write('\r' + ' ' * self.terminal_width)
        sys.stdout.write(status)
        sys.stdout.flush()
        
    def finish(self, message: Optional[str] = None):
        """Finish progress indication."""
        elapsed = time.time() - self.start_time
        
        # Clear current line
        sys.stdout.write('\r' + ' ' * self.terminal_width + '\r')
        
        # Print completion message
        if message:
            print(f"{self.COLORS['GREEN']}✓{self.COLORS['RESET']} {message}")
        else:
            print(f"{self.COLORS['GREEN']}✓{self.COLORS['RESET']} Complete!")
            
        # Print statistics
        self._print_summary(elapsed)
        
    def error(self, message: str):
        """Show error message."""
        sys.stdout.write('\r' + ' ' * self.terminal_width + '\r')
        print(f"{self.COLORS['RED']}✗{self.COLORS['RESET']} {message}")
        
    def _print_summary(self, elapsed: float):
        """Print final summary statistics."""
        print(f"\n{self.COLORS['BOLD']}Summary:{self.COLORS['RESET']}")
        print(f"  • Time elapsed: {self._format_time(elapsed)}")
        
        if self.stats.get('rows_processed'):
            rate = self.stats['rows_processed'] / elapsed if elapsed > 0 else 0
            print(f"  • Rows processed: {self.COLORS['GREEN']}{self.stats['rows_processed']:,}{self.COLORS['RESET']} ({rate:.0f}/sec)")
            
        if self.stats.get('users_processed'):
            user_rate = self.stats['users_processed'] / elapsed if elapsed > 0 else 0
            print(f"  • Users processed: {self.COLORS['CYAN']}{self.stats['users_processed']:,}{self.COLORS['RESET']} ({user_rate:.1f}/sec)")
            
            # Calculate average readings per user
            if self.stats.get('rows_processed'):
                avg_per_user = self.stats['rows_processed'] / self.stats['users_processed']
                print(f"  • Avg readings/user: {avg_per_user:.0f}")
            
        if self.stats.get('accepted') or self.stats.get('rejected'):
            total = self.stats.get('accepted', 0) + self.stats.get('rejected', 0)
            if total > 0:
                acceptance = (self.stats.get('accepted', 0) / total) * 100
                if acceptance >= 90:
                    acc_color = self.COLORS['GREEN']
                elif acceptance >= 70:
                    acc_color = self.COLORS['YELLOW']
                else:
                    acc_color = self.COLORS['RED']
                print(f"  • Acceptance rate: {acc_color}{acceptance:.1f}%{self.COLORS['RESET']}")
                print(f"    - Accepted: {self.stats.get('accepted', 0):,}")
                print(f"    - Rejected: {self.stats.get('rejected', 0):,}")
                
        if self.stats.get('errors'):
            print(f"  • {self.COLORS['RED']}Errors: {self.stats['errors']}{self.COLORS['RESET']}")
            
    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            

class ProcessingStatus:
    """Simple status messages with color and icons."""
    
    @staticmethod
    def info(message: str):
        """Print info message with icon."""
        print(f"\033[94mℹ\033[0m  {message}")
        
    @staticmethod
    def success(message: str):
        """Print success message with icon."""
        print(f"\033[92m✓\033[0m  {message}")
        
    @staticmethod
    def warning(message: str):
        """Print warning message with icon."""
        print(f"\033[93m⚠\033[0m  {message}")
        
    @staticmethod
    def error(message: str):
        """Print error message with icon."""
        print(f"\033[91m✗\033[0m  {message}")
        
    @staticmethod
    def processing(message: str):
        """Print processing message with icon."""
        print(f"\033[96m⟳\033[0m  {message}")
        
    @staticmethod
    def section(title: str):
        """Print section header."""
        width = shutil.get_terminal_size(fallback=(80, 20)).columns
        separator = "=" * min(60, width - 1)
        print(f"\n\033[1m{separator}\033[0m")
        print(f"\033[1m{title.upper()}\033[0m")
        print(f"\033[1m{separator}\033[0m")
        

def count_csv_lines(filepath: str) -> Optional[int]:
    """Quickly count lines in CSV file for progress bar."""
    try:
        with open(filepath, 'r') as f:
            # Count lines (subtract 1 for header)
            return sum(1 for _ in f) - 1
    except:
        return None