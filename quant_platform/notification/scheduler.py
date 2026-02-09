"""
Alert scheduler module for periodic strategy checks.
"""

import json
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from config.settings import get_settings


@dataclass
class ScheduleConfig:
    """Scheduler configuration."""
    enabled: bool = False
    frequency: str = "daily"  # daily, weekly
    check_time: str = "09:30"
    last_run: str = ""
    
    def should_run_now(self) -> bool:
        """Check if scheduler should run based on current time."""
        if not self.enabled:
            return False
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        today = now.strftime("%Y-%m-%d")
        
        # Check if already run today
        if self.last_run == today:
            return False
        
        # Check if it's time to run
        if current_time >= self.check_time:
            return True
        
        return False


class AlertScheduler:
    """
    Scheduler for periodic strategy alert checks.
    Runs in background thread and triggers strategy evaluation.
    """
    
    def __init__(self):
        """Initialize scheduler."""
        self.settings = get_settings()
        self.config_file = self.settings.base_dir / "data" / "scheduler_config.json"
        self.lock_file = self.settings.base_dir / "data" / "scheduler.lock"
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_callbacks: List[Callable] = []
        
        # Ensure data directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> ScheduleConfig:
        """Load scheduler configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return ScheduleConfig(
                    enabled=data.get('enabled', False),
                    frequency=data.get('frequency', 'daily'),
                    check_time=data.get('check_time', '09:30'),
                    last_run=data.get('last_run', ''),
                )
            except Exception:
                pass
        return ScheduleConfig()
    
    def save_config(self, config: ScheduleConfig) -> bool:
        """Save scheduler configuration."""
        try:
            data = {
                'enabled': config.enabled,
                'frequency': config.frequency,
                'check_time': config.check_time,
                'last_run': config.last_run,
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception:
            return False
    
    def update_last_run(self):
        """Update last run timestamp."""
        config = self.load_config()
        config.last_run = datetime.now().strftime("%Y-%m-%d")
        self.save_config(config)
    
    def add_check_callback(self, callback: Callable):
        """
        Add a callback function to be called during scheduled checks.
        
        Args:
            callback: Function to call (should return Dict with results)
        """
        self._check_callbacks.append(callback)
    
    def remove_check_callback(self, callback: Callable):
        """Remove a callback function."""
        if callback in self._check_callbacks:
            self._check_callbacks.remove(callback)
    
    def _acquire_lock(self) -> bool:
        """Acquire scheduler lock to prevent multiple instances."""
        try:
            if self.lock_file.exists():
                # Check if lock is stale (older than 1 hour)
                mtime = datetime.fromtimestamp(self.lock_file.stat().st_mtime)
                if (datetime.now() - mtime).total_seconds() > 3600:
                    self.lock_file.unlink()
                else:
                    return False
            
            self.lock_file.write_text(str(datetime.now()))
            return True
        except Exception:
            return False
    
    def _release_lock(self):
        """Release scheduler lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception:
            pass
    
    def _run_checks(self) -> Dict[str, Any]:
        """
        Run all registered check callbacks.
        
        Returns:
            Dictionary with results from all callbacks
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': [],
        }
        
        for callback in self._check_callbacks:
            try:
                result = callback()
                results['checks'].append({
                    'callback': callback.__name__,
                    'success': True,
                    'result': result,
                })
            except Exception as e:
                results['checks'].append({
                    'callback': callback.__name__,
                    'success': False,
                    'error': str(e),
                })
        
        return results
    
    def _scheduler_loop(self):
        """Main scheduler loop running in background thread."""
        while self._running:
            try:
                config = self.load_config()
                
                if config.should_run_now():
                    # Run checks
                    results = self._run_checks()
                    
                    # Update last run
                    self.update_last_run()
                    
                    # Log results (you could also save to file)
                    print(f"Scheduler ran at {results['timestamp']}")
                
            except Exception as e:
                print(f"Scheduler error: {e}")
            
            # Sleep for 60 seconds before next check
            for _ in range(60):
                if not self._running:
                    break
                time.sleep(1)
    
    def start(self) -> bool:
        """
        Start the scheduler background thread.
        
        Returns:
            True if started successfully
        """
        if self._running:
            return True
        
        if not self._acquire_lock():
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop(self):
        """Stop the scheduler background thread."""
        self._running = False
        self._release_lock()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
    
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
    
    def run_now(self) -> Dict[str, Any]:
        """
        Manually trigger a check run immediately.
        
        Returns:
            Results dictionary
        """
        return self._run_checks()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Status dictionary
        """
        config = self.load_config()
        
        return {
            'enabled': config.enabled,
            'running': self._running,
            'frequency': config.frequency,
            'check_time': config.check_time,
            'last_run': config.last_run,
            'callbacks_count': len(self._check_callbacks),
        }
