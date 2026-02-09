"""
Technical indicators calculation module.
Provides common technical analysis indicators for strategy development.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class TechnicalIndicators:
    """
    Collection of technical indicators for trading strategies.
    All methods are designed to work with pandas Series/DataFrame.
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            SMA series
        """
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            data: Price series
            period: Number of periods (span)
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill initial NaN with neutral value
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with 'macd', 'signal', 'histogram' columns
        """
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with 'upper', 'middle', 'lower' columns
        """
        middle = data.rolling(window=period, min_periods=1).mean()
        std = data.rolling(window=period, min_periods=1).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def atr_from_close(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Simplified ATR calculation using only close prices.
        Uses price range as proxy for true range.
        
        Args:
            data: Close price series
            period: ATR period
            
        Returns:
            Approximate ATR series
        """
        # Use absolute daily returns as proxy
        returns = data.pct_change().abs()
        atr = returns.ewm(span=period, adjust=False).mean() * data
        
        return atr
    
    @staticmethod
    def volatility(data: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Rolling volatility (standard deviation of returns).
        
        Args:
            data: Price series
            period: Rolling window
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Volatility series
        """
        returns = data.pct_change()
        vol = returns.rolling(window=period, min_periods=1).std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Price momentum (rate of change).
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            Momentum series (percentage change)
        """
        return data.pct_change(periods=period) * 100
    
    @staticmethod
    def drawdown(data: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdown series.
        
        Args:
            data: Price/value series
            
        Returns:
            DataFrame with 'peak', 'drawdown', 'drawdown_pct' columns
        """
        peak = data.cummax()
        drawdown = data - peak
        drawdown_pct = (data / peak - 1) * 100
        
        return pd.DataFrame({
            'peak': peak,
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct
        })
    
    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossover events (series1 crosses above series2).
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Boolean series (True when crossover occurs)
        """
        prev_diff = (series1.shift(1) - series2.shift(1))
        curr_diff = (series1 - series2)
        
        return (prev_diff <= 0) & (curr_diff > 0)
    
    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossunder events (series1 crosses below series2).
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Boolean series (True when crossunder occurs)
        """
        prev_diff = (series1.shift(1) - series2.shift(1))
        curr_diff = (series1 - series2)
        
        return (prev_diff >= 0) & (curr_diff < 0)
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with 'k' and 'd' columns
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        
        k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d = k.rolling(window=d_period, min_periods=1).mean()
        
        return pd.DataFrame({'k': k.fillna(50), 'd': d.fillna(50)})
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period
            
        Returns:
            Williams %R series (-100 to 0)
        """
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        wr = ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        return wr.fillna(-50)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price (cumulative).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
