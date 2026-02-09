"""
Performance metrics calculation for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    risk_free_rate: float = 0.03      # Annual risk-free rate (3%)
    trading_days_per_year: int = 252  # Trading days for annualization
    

class PerformanceMetrics:
    """
    Calculate various performance metrics for backtest results.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
    
    def calculate_all(
        self,
        portfolio_values: pd.Series,
        benchmark_values: pd.Series = None,
        trades: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            portfolio_values: Daily portfolio value series
            benchmark_values: Optional benchmark for comparison
            trades: List of trades for win rate calculation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic returns
        metrics['Total Return (%)'] = self.total_return(portfolio_values)
        metrics['CAGR (%)'] = self.cagr(portfolio_values)
        
        # Risk metrics
        metrics['Max Drawdown (%)'] = self.max_drawdown(portfolio_values)
        metrics['Max DD Duration (Days)'] = self.max_drawdown_duration(portfolio_values)
        metrics['Volatility (%)'] = self.volatility(portfolio_values)
        
        # Risk-adjusted returns
        metrics['Sharpe Ratio'] = self.sharpe_ratio(portfolio_values)
        metrics['Sortino Ratio'] = self.sortino_ratio(portfolio_values)
        metrics['Calmar Ratio'] = self.calmar_ratio(portfolio_values)
        
        # Trade statistics
        if trades:
            trade_stats = self.trade_statistics(trades)
            metrics.update(trade_stats)
        
        # Benchmark comparison
        if benchmark_values is not None:
            metrics['Alpha (%)'] = self.alpha(portfolio_values, benchmark_values)
            metrics['Beta'] = self.beta(portfolio_values, benchmark_values)
            metrics['Information Ratio'] = self.information_ratio(portfolio_values, benchmark_values)
        
        return metrics
    
    def total_return(self, values: pd.Series) -> float:
        """Calculate total return percentage."""
        if len(values) < 2:
            return 0.0
        return (values.iloc[-1] / values.iloc[0] - 1) * 100
    
    def cagr(self, values: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(values) < 2:
            return 0.0
        
        total_return = values.iloc[-1] / values.iloc[0]
        days = (values.index[-1] - values.index[0]).days
        
        if days <= 0:
            return 0.0
        
        years = days / 365.25
        cagr = (total_return ** (1 / years) - 1) * 100
        
        return cagr
    
    def max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(values) < 2:
            return 0.0
        
        rolling_max = values.cummax()
        drawdown = (values / rolling_max - 1) * 100
        
        return drawdown.min()
    
    def max_drawdown_duration(self, values: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(values) < 2:
            return 0
        
        rolling_max = values.cummax()
        is_peak = values == rolling_max
        
        # Find peak dates
        peak_dates = pd.Series(values.index, index=values.index).where(is_peak).ffill()
        
        # Calculate duration from last peak
        duration = values.index - peak_dates
        
        # Get max duration in days
        max_duration = duration.max()
        
        return max_duration.days if hasattr(max_duration, 'days') else 0
    
    def volatility(self, values: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(values) < 2:
            return 0.0
        
        returns = values.pct_change().dropna()
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(self.config.trading_days_per_year)
        
        return vol * 100
    
    def sharpe_ratio(self, values: pd.Series) -> float:
        """Calculate Sharpe Ratio."""
        if len(values) < 2:
            return 0.0
        
        returns = values.pct_change().dropna()
        
        if returns.std() == 0:
            return 0.0
        
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = returns - rf_daily
        
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(self.config.trading_days_per_year)
        
        return sharpe
    
    def sortino_ratio(self, values: pd.Series) -> float:
        """Calculate Sortino Ratio (downside risk adjusted)."""
        if len(values) < 2:
            return 0.0
        
        returns = values.pct_change().dropna()
        
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = returns - rf_daily
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std() * np.sqrt(self.config.trading_days_per_year)
        
        annual_excess = excess_returns.mean() * self.config.trading_days_per_year
        
        return annual_excess / downside_std
    
    def calmar_ratio(self, values: pd.Series) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
        cagr = self.cagr(values)
        max_dd = abs(self.max_drawdown(values))
        
        if max_dd == 0:
            return 0.0
        
        return cagr / max_dd
    
    def alpha(self, portfolio: pd.Series, benchmark: pd.Series) -> float:
        """Calculate Jensen's Alpha (annualized)."""
        if len(portfolio) < 2 or len(benchmark) < 2:
            return 0.0
        
        # Align indices
        common_idx = portfolio.index.intersection(benchmark.index)
        port_returns = portfolio.loc[common_idx].pct_change().dropna()
        bench_returns = benchmark.loc[common_idx].pct_change().dropna()
        
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        
        # Calculate beta
        covar = port_returns.cov(bench_returns)
        bench_var = bench_returns.var()
        
        if bench_var == 0:
            return 0.0
        
        beta = covar / bench_var
        
        # Jensen's Alpha
        port_excess = (port_returns - rf_daily).mean()
        bench_excess = (bench_returns - rf_daily).mean()
        
        alpha_daily = port_excess - beta * bench_excess
        alpha_annual = alpha_daily * self.config.trading_days_per_year * 100
        
        return alpha_annual
    
    def beta(self, portfolio: pd.Series, benchmark: pd.Series) -> float:
        """Calculate portfolio beta."""
        if len(portfolio) < 2 or len(benchmark) < 2:
            return 1.0
        
        # Align indices
        common_idx = portfolio.index.intersection(benchmark.index)
        port_returns = portfolio.loc[common_idx].pct_change().dropna()
        bench_returns = benchmark.loc[common_idx].pct_change().dropna()
        
        bench_var = bench_returns.var()
        
        if bench_var == 0:
            return 1.0
        
        covar = port_returns.cov(bench_returns)
        
        return covar / bench_var
    
    def information_ratio(self, portfolio: pd.Series, benchmark: pd.Series) -> float:
        """Calculate Information Ratio."""
        if len(portfolio) < 2 or len(benchmark) < 2:
            return 0.0
        
        # Align indices
        common_idx = portfolio.index.intersection(benchmark.index)
        port_returns = portfolio.loc[common_idx].pct_change().dropna()
        bench_returns = benchmark.loc[common_idx].pct_change().dropna()
        
        # Tracking difference
        tracking_diff = port_returns - bench_returns
        
        if tracking_diff.std() == 0:
            return 0.0
        
        ir = (tracking_diff.mean() / tracking_diff.std()) * np.sqrt(self.config.trading_days_per_year)
        
        return ir
    
    def trade_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate trade-level statistics.
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
            
        Returns:
            Trade statistics
        """
        if not trades:
            return {}
        
        # Extract P&L from trades
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        
        if not pnls:
            return {'Trade Count': len(trades)}
        
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]
        
        stats = {
            'Trade Count': len(trades),
            'Win Rate (%)': (len(winning) / len(pnls)) * 100 if pnls else 0,
        }
        
        if winning:
            stats['Avg Win (%)'] = np.mean(winning) * 100
        
        if losing:
            stats['Avg Loss (%)'] = np.mean(losing) * 100
        
        if winning and losing:
            stats['Profit Factor'] = abs(sum(winning) / sum(losing)) if sum(losing) != 0 else 0
        
        return stats
    
    def drawdown_series(self, values: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = values.cummax()
        drawdown = (values / rolling_max - 1) * 100
        return drawdown
    
    def rolling_sharpe(self, values: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        returns = values.pct_change()
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess = returns - rf_daily
        
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.config.trading_days_per_year)
        
        return rolling_sharpe
    
    def monthly_returns(self, values: pd.Series) -> pd.Series:
        """Calculate monthly returns."""
        monthly = values.resample('ME').last()
        returns = monthly.pct_change() * 100
        return returns
    
    def yearly_returns(self, values: pd.Series) -> pd.Series:
        """Calculate yearly returns."""
        yearly = values.resample('YE').last()
        returns = yearly.pct_change() * 100
        return returns
