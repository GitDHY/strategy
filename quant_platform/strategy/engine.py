"""
Strategy engine for executing user-defined strategies.
Provides the StrategyContext API and orchestrates strategy execution.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

from config.settings import get_settings
from data.fetcher import DataFetcher, get_data_fetcher
from data.indicators import TechnicalIndicators
from strategy.sandbox import SafeExecutor, StrategyError


@dataclass
class StrategyResult:
    """Result of strategy execution."""
    success: bool
    target_weights: Dict[str, float]
    message: str = ""
    signals: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class StrategyContext:
    """
    Strategy execution context.
    Provides API for user strategies to access data and set allocations.
    """
    
    def __init__(
        self,
        tickers: List[str],
        current_weights: Dict[str, float],
        current_date: date,
        data_fetcher: Optional[DataFetcher] = None,
        lookback_days: int = 252,
    ):
        """
        Initialize strategy context.
        
        Args:
            tickers: List of available tickers
            current_weights: Current portfolio weights (0-100)
            current_date: Current simulation date
            data_fetcher: Data fetcher instance
            lookback_days: Days of historical data available
        """
        self._tickers = tickers
        self._current_weights = current_weights.copy()
        self._target_weights: Optional[Dict[str, float]] = None
        self._current_date = current_date
        self._lookback_days = lookback_days
        self._data_fetcher = data_fetcher or get_data_fetcher()
        self._indicators = TechnicalIndicators()
        self._signals: List[str] = []
        
        # Cache for price data
        self._price_cache: Dict[str, pd.Series] = {}
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
    
    @property
    def tickers(self) -> List[str]:
        """Available tickers."""
        return self._tickers.copy()
    
    @property
    def current_date(self) -> date:
        """Current simulation date."""
        return self._current_date
    
    @property
    def signals(self) -> List[str]:
        """Generated signals/messages."""
        return self._signals.copy()
    
    def log(self, message: str):
        """Log a signal or message."""
        self._signals.append(message)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        return self._current_weights.copy()
    
    def set_target_weights(self, weights: Dict[str, float]):
        """
        Set target portfolio weights.
        
        Args:
            weights: Dictionary of ticker -> weight (percentage 0-100)
                     Unknown tickers are ignored with a warning logged.
        """
        # Filter to only include known tickers, warn about unknown ones
        filtered_weights = {}
        for ticker, weight in weights.items():
            if ticker in self._tickers:
                filtered_weights[ticker] = weight
            else:
                self._signals.append(f"⚠️ 忽略未知标的: {ticker}")
        
        self._target_weights = filtered_weights
    
    def get_target_weights(self) -> Optional[Dict[str, float]]:
        """Get target weights if set."""
        return self._target_weights
    
    def _get_price_data(self, ticker: str) -> pd.Series:
        """Get cached price data for ticker."""
        if ticker not in self._price_cache:
            end_date = self._current_date.strftime('%Y-%m-%d')
            df = self._data_fetcher.fetch_prices(
                ticker,
                end_date=end_date,
                lookback_days=self._lookback_days
            )
            if not df.empty:
                self._price_cache[ticker] = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            else:
                self._price_cache[ticker] = pd.Series(dtype=float)
        
        return self._price_cache[ticker]
    
    def get_price(self, ticker: str, lookback: int = None) -> pd.Series:
        """
        Get price series for a ticker.
        
        Args:
            ticker: Ticker symbol
            lookback: Number of days (default: all available)
            
        Returns:
            Price series
        """
        prices = self._get_price_data(ticker)
        
        if lookback and len(prices) > lookback:
            return prices.iloc[-lookback:]
        
        return prices
    
    def get_prices(self, tickers: List[str] = None, lookback: int = None) -> pd.DataFrame:
        """
        Get price DataFrame for multiple tickers.
        
        Args:
            tickers: List of tickers (default: all)
            lookback: Number of days
            
        Returns:
            DataFrame with tickers as columns
        """
        if tickers is None:
            tickers = self._tickers
        
        data = {}
        for ticker in tickers:
            prices = self.get_price(ticker, lookback)
            if not prices.empty:
                data[ticker] = prices
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df.ffill().bfill()
    
    def get_returns(self, ticker: str, lookback: int = None) -> pd.Series:
        """
        Get return series for a ticker.
        
        Args:
            ticker: Ticker symbol
            lookback: Number of days
            
        Returns:
            Daily return series
        """
        prices = self.get_price(ticker, lookback)
        return prices.pct_change().fillna(0)
    
    def vix(self, lookback: int = None) -> pd.Series:
        """
        Get VIX (Volatility Index) series.
        
        Args:
            lookback: Number of days
            
        Returns:
            VIX series
        """
        return self.get_price("^VIX", lookback)
    
    def current_vix(self) -> float:
        """Get current VIX value."""
        vix = self.vix(5)
        return vix.iloc[-1] if not vix.empty else 20.0
    
    # Technical Indicators
    def ma(self, ticker: str, period: int) -> pd.Series:
        """Simple Moving Average."""
        prices = self.get_price(ticker)
        return self._indicators.sma(prices, period)
    
    def ema(self, ticker: str, period: int) -> pd.Series:
        """Exponential Moving Average."""
        prices = self.get_price(ticker)
        return self._indicators.ema(prices, period)
    
    def rsi(self, ticker: str, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        prices = self.get_price(ticker)
        return self._indicators.rsi(prices, period)
    
    def macd(self, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD indicator."""
        prices = self.get_price(ticker)
        return self._indicators.macd(prices, fast, slow, signal)
    
    def bollinger(self, ticker: str, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands."""
        prices = self.get_price(ticker)
        return self._indicators.bollinger_bands(prices, period, std)
    
    def atr(self, ticker: str, period: int = 14) -> pd.Series:
        """Average True Range (simplified using close prices)."""
        prices = self.get_price(ticker)
        return self._indicators.atr_from_close(prices, period)
    
    def volatility(self, ticker: str, period: int = 20, annualize: bool = True) -> pd.Series:
        """Rolling volatility."""
        prices = self.get_price(ticker)
        return self._indicators.volatility(prices, period, annualize)
    
    def momentum(self, ticker: str, period: int = 10) -> pd.Series:
        """Price momentum."""
        prices = self.get_price(ticker)
        return self._indicators.momentum(prices, period)
    
    def drawdown(self, ticker: str) -> pd.DataFrame:
        """Drawdown analysis."""
        prices = self.get_price(ticker)
        return self._indicators.drawdown(prices)
    
    # Utility methods
    def current_price(self, ticker: str) -> float:
        """Get current price for ticker."""
        prices = self.get_price(ticker, 5)
        return prices.iloc[-1] if not prices.empty else 0.0
    
    def price_above_ma(self, ticker: str, period: int) -> bool:
        """Check if price is above moving average."""
        price = self.current_price(ticker)
        ma = self.ma(ticker, period)
        return price > ma.iloc[-1] if not ma.empty else False
    
    def price_below_ma(self, ticker: str, period: int) -> bool:
        """Check if price is below moving average."""
        return not self.price_above_ma(ticker, period)
    
    def ma_cross_up(self, ticker: str, short_period: int, long_period: int) -> bool:
        """Check if short MA crossed above long MA recently."""
        short_ma = self.ma(ticker, short_period)
        long_ma = self.ma(ticker, long_period)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return False
        
        return (short_ma.iloc[-2] <= long_ma.iloc[-2]) and (short_ma.iloc[-1] > long_ma.iloc[-1])
    
    def ma_cross_down(self, ticker: str, short_period: int, long_period: int) -> bool:
        """Check if short MA crossed below long MA recently."""
        short_ma = self.ma(ticker, short_period)
        long_ma = self.ma(ticker, long_period)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return False
        
        return (short_ma.iloc[-2] >= long_ma.iloc[-2]) and (short_ma.iloc[-1] < long_ma.iloc[-1])


class StrategyEngine:
    """
    Main strategy execution engine.
    Handles strategy storage, validation, and execution.
    """
    
    def __init__(self):
        """Initialize strategy engine."""
        self.settings = get_settings()
        self.storage_path = self.settings.strategies_file
        self.executor = SafeExecutor(timeout_seconds=self.settings.strategy_timeout_seconds)
        
        # Ensure data directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._strategies: Dict[str, dict] = {}
        self._loaded = False
    
    def _ensure_loaded(self):
        """Load strategies from disk if not already loaded."""
        if not self._loaded:
            self.load()
    
    def load(self) -> Dict[str, dict]:
        """Load strategies from storage."""
        self._strategies = {}
        
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self._strategies = json.load(f)
            except Exception as e:
                print(f"Error loading strategies: {e}")
        
        self._loaded = True
        return self._strategies
    
    def save(self) -> bool:
        """Save strategies to storage."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._strategies, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving strategies: {e}")
            return False
    
    def get_all(self) -> Dict[str, dict]:
        """Get all saved strategies."""
        self._ensure_loaded()
        return self._strategies.copy()
    
    def get(self, name: str) -> Optional[dict]:
        """Get a strategy by name."""
        self._ensure_loaded()
        return self._strategies.get(name)
    
    def save_strategy(
        self,
        name: str,
        code: str,
        description: str = "",
        portfolio_name: str = ""
    ) -> bool:
        """
        Save a strategy.
        
        Args:
            name: Strategy name
            code: Strategy Python code
            description: Strategy description
            portfolio_name: Associated portfolio
            
        Returns:
            True if successful
        """
        self._ensure_loaded()
        
        self._strategies[name] = {
            'name': name,
            'code': code,
            'description': description,
            'portfolio_name': portfolio_name,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        
        return self.save()
    
    def delete_strategy(self, name: str) -> bool:
        """Delete a strategy."""
        self._ensure_loaded()
        
        if name in self._strategies:
            del self._strategies[name]
            return self.save()
        
        return False
    
    def validate_strategy(self, code: str) -> Dict[str, Any]:
        """
        Validate strategy code.
        
        Args:
            code: Strategy Python code
            
        Returns:
            Validation result dictionary
        """
        return self.executor.validate_code(code)
    
    def execute(
        self,
        code: str,
        tickers: List[str],
        current_weights: Dict[str, float],
        current_date: date = None,
        lookback_days: int = 252,
    ) -> StrategyResult:
        """
        Execute a strategy.
        
        Args:
            code: Strategy Python code
            tickers: Available tickers
            current_weights: Current portfolio weights
            current_date: Simulation date
            lookback_days: Historical data days
            
        Returns:
            StrategyResult
        """
        import time
        start_time = time.time()
        
        if current_date is None:
            current_date = date.today()
        
        # Create context
        ctx = StrategyContext(
            tickers=tickers,
            current_weights=current_weights,
            current_date=current_date,
            lookback_days=lookback_days,
        )
        
        # Create execution context with API
        execution_context = {
            'ctx': ctx,
            # Expose numpy and pandas for calculations
            'np': np,
            'pd': pd,
            # Expose common math functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
        }
        
        try:
            # Execute strategy
            self.executor.execute(code, execution_context)
            
            # Get results
            target_weights = ctx.get_target_weights()
            
            if target_weights is None:
                # No change - use current weights
                target_weights = current_weights.copy()
            
            execution_time = time.time() - start_time
            
            return StrategyResult(
                success=True,
                target_weights=target_weights,
                message="Strategy executed successfully",
                signals=ctx.signals,
                execution_time=execution_time,
            )
            
        except StrategyError as e:
            return StrategyResult(
                success=False,
                target_weights=current_weights.copy(),
                message=str(e),
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                target_weights=current_weights.copy(),
                message=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
            )
    
    def run_strategy_check(
        self,
        strategy_name: str,
        tickers: List[str],
        current_weights: Dict[str, float],
    ) -> StrategyResult:
        """
        Run a saved strategy check.
        
        Args:
            strategy_name: Name of saved strategy
            tickers: Portfolio tickers
            current_weights: Current weights
            
        Returns:
            StrategyResult
        """
        strategy = self.get(strategy_name)
        
        if strategy is None:
            return StrategyResult(
                success=False,
                target_weights=current_weights,
                message=f"Strategy '{strategy_name}' not found",
            )
        
        return self.execute(
            code=strategy['code'],
            tickers=tickers,
            current_weights=current_weights,
        )
