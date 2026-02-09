"""
Global settings and configuration management for Quant Platform.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class BacktestDefaults:
    """Default parameters for backtesting."""
    initial_capital: float = 100000.0
    rebalance_freq: str = "monthly"  # daily, weekly, monthly
    commission_fixed: float = 0.0     # Fixed cost per trade
    commission_pct: float = 0.001     # Percentage commission (0.1%)
    slippage_pct: float = 0.001       # Slippage (0.1%)
    risk_free_rate: float = 0.03      # Annual risk-free rate (3%)


@dataclass
class NotificationDefaults:
    """Default notification settings."""
    # Email (SMTP)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_from: str = ""
    email_to: str = ""
    email_pwd: str = ""
    
    # Server酱 / PushPlus
    serverchan_key: str = ""  # Server酱 SendKey
    pushplus_token: str = ""  # PushPlus Token
    
    # Scheduling
    check_frequency: str = "daily"  # daily, weekly
    check_time: str = "09:30"


@dataclass
class Settings:
    """Main settings class for Quant Platform."""
    
    # Directory paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_cache_dir: Path = field(default=None)
    
    # Parent project paths (for importing existing data)
    parent_project_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    # Default configurations
    backtest: BacktestDefaults = field(default_factory=BacktestDefaults)
    notification: NotificationDefaults = field(default_factory=NotificationDefaults)
    
    # Strategy execution
    strategy_timeout_seconds: int = 10  # Max execution time for user strategies
    max_lookback_days: int = 1260       # Max historical data (~5 years)
    
    # Data cache settings
    cache_expiry_hours: int = 4         # Price data cache expiry
    
    def __post_init__(self):
        """Initialize computed paths after creation."""
        if self.data_cache_dir is None:
            self.data_cache_dir = self.base_dir / "data" / "cache"
        
        # Ensure directories exist
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def portfolios_file(self) -> Path:
        """Path to portfolios JSON file (new location)."""
        return self.base_dir / "data" / "portfolios.json"
    
    @property
    def strategies_file(self) -> Path:
        """Path to saved strategies JSON file."""
        return self.base_dir / "data" / "strategies.json"
    
    @property
    def notification_config_file(self) -> Path:
        """Path to notification config file."""
        return self.base_dir / "data" / "notification_config.json"
    
    @property
    def legacy_portfolios_file(self) -> Path:
        """Path to existing portfolios.json in parent project."""
        return self.parent_project_dir / "portfolios.json"
    
    @property
    def legacy_alert_config_file(self) -> Path:
        """Path to existing alert_config.json in parent project."""
        return self.parent_project_dir / "alert_config.json"
    
    def load_notification_config(self) -> NotificationDefaults:
        """Load notification config from file, falling back to legacy config."""
        config = NotificationDefaults()
        
        # Try new config file first
        if self.notification_config_file.exists():
            try:
                with open(self.notification_config_file, 'r') as f:
                    data = json.load(f)
                    config.smtp_server = data.get('smtp_server', config.smtp_server)
                    config.smtp_port = data.get('smtp_port', config.smtp_port)
                    config.email_from = data.get('email_from', config.email_from)
                    config.email_to = data.get('email_to', config.email_to)
                    config.email_pwd = data.get('email_pwd', config.email_pwd)
                    config.serverchan_key = data.get('serverchan_key', config.serverchan_key)
                    config.pushplus_token = data.get('pushplus_token', config.pushplus_token)
                    config.check_frequency = data.get('check_frequency', config.check_frequency)
                    config.check_time = data.get('check_time', config.check_time)
                return config
            except Exception:
                pass
        
        # Try legacy alert_config.json
        if self.legacy_alert_config_file.exists():
            try:
                with open(self.legacy_alert_config_file, 'r') as f:
                    data = json.load(f)
                    config.smtp_server = data.get('smtp_server', config.smtp_server)
                    config.smtp_port = data.get('smtp_port', config.smtp_port)
                    config.email_from = data.get('email_from', config.email_from)
                    config.email_to = data.get('email_to', config.email_to)
                    config.email_pwd = data.get('email_pwd', config.email_pwd)
                    # Legacy may have wechat webhook
                    channels = data.get('channels', {})
                    if channels.get('wechat_webhook'):
                        # Store for reference but we'll use Server酱/PushPlus
                        pass
            except Exception:
                pass
        
        return config
    
    def save_notification_config(self, config: NotificationDefaults) -> bool:
        """Save notification config to file."""
        try:
            self.notification_config_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'smtp_server': config.smtp_server,
                'smtp_port': config.smtp_port,
                'email_from': config.email_from,
                'email_to': config.email_to,
                'email_pwd': config.email_pwd,
                'serverchan_key': config.serverchan_key,
                'pushplus_token': config.pushplus_token,
                'check_frequency': config.check_frequency,
                'check_time': config.check_time,
            }
            with open(self.notification_config_file, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception:
            return False


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
