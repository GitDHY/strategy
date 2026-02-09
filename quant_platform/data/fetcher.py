"""
Data fetcher module with yfinance wrapper and local caching.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Union

import pandas as pd
import yfinance as yf

from config.settings import get_settings


class DataFetcher:
    """
    Market data fetcher with caching support.
    Wraps yfinance API and provides local file caching.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data fetcher.
        
        Args:
            cache_dir: Directory for cached data files
        """
        settings = get_settings()
        self.cache_dir = cache_dir or settings.data_cache_dir
        self.cache_expiry_hours = settings.cache_expiry_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, tickers: List[str], start: str, end: str) -> Path:
        """Generate cache file path based on request parameters."""
        # Create a unique hash for the request
        key = f"{sorted(tickers)}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"prices_{hash_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False
        
        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        return age_hours < self.cache_expiry_hours
    
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize yfinance DataFrame to extract adjusted close prices.
        Handles different yfinance versions and column formats.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            level_values_0 = df.columns.get_level_values(0).unique().tolist()
            level_values_1 = df.columns.get_level_values(1).unique().tolist() if df.columns.nlevels > 1 else []
            
            price_types = ['Adj Close', 'Close', 'Price']
            
            for price_type in price_types:
                if price_type in level_values_0:
                    return df[price_type]
                elif price_type in level_values_1:
                    df_swapped = df.swaplevel(axis=1)
                    return df_swapped[price_type]
            
            # Fallback: return first level
            try:
                return df[level_values_0[0]]
            except Exception:
                pass
            
            return df
        
        # Single-level columns
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        if 'Close' in df.columns:
            return df['Close']
        if 'Price' in df.columns:
            return df['Price']
        
        return df
    
    def fetch_prices(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 252,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            lookback_days: If start_date not provided, look back this many days
            use_cache: Whether to use local cache
            
        Returns:
            DataFrame with tickers as columns and dates as index
        """
        # Normalize tickers to list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Check cache
        cache_path = self._get_cache_path(tickers, start_date, end_date)
        
        if use_cache and self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # Verify all tickers are present
                if all(t in df.columns for t in tickers):
                    return df[tickers]
            except Exception:
                pass
        
        # Fetch from yfinance
        try:
            raw_data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if raw_data is None or raw_data.empty:
                return pd.DataFrame()
            
            # Normalize to get adjusted close prices
            prices = self._normalize_prices(raw_data)
            
            # Handle single ticker case
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
            elif len(tickers) == 1 and isinstance(prices, pd.DataFrame):
                # For single ticker, ensure we have exactly one column with the ticker name
                if prices.shape[1] == 1:
                    # Single column - rename it to ticker name
                    if tickers[0] not in prices.columns:
                        prices.columns = tickers
                elif prices.shape[1] > 1:
                    # Multiple columns - try to find the right one or take first
                    if tickers[0] in prices.columns:
                        prices = prices[[tickers[0]]]
                    else:
                        # Take first column and rename to ticker
                        prices = prices.iloc[:, [0]]
                        prices.columns = tickers
            
            # Clean data
            prices = prices.dropna(axis=1, how='all')
            prices = prices.ffill().bfill()
            
            # Cache the result
            if use_cache and not prices.empty:
                try:
                    prices.to_parquet(cache_path)
                except Exception:
                    pass
            
            return prices
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_vix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 252
    ) -> pd.Series:
        """
        Fetch VIX (Volatility Index) data.
        
        Returns:
            Series with VIX values
        """
        df = self.fetch_prices(
            "^VIX",
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days
        )
        
        if df.empty:
            return pd.Series(dtype=float)
        
        return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Fetch full OHLCV data for a single ticker.
        
        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        try:
            raw_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if raw_data is None or raw_data.empty:
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Flatten to just price type
                raw_data.columns = raw_data.columns.get_level_values(0)
            
            # Select OHLCV columns
            columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            available = [c for c in columns if c in raw_data.columns]
            
            return raw_data[available].ffill().bfill()
            
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data files."""
        for f in self.cache_dir.glob("*.parquet"):
            try:
                f.unlink()
            except Exception:
                pass


# Singleton instance
_data_fetcher: Optional[DataFetcher] = None


def get_data_fetcher() -> DataFetcher:
    """Get the global DataFetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = DataFetcher()
    return _data_fetcher
