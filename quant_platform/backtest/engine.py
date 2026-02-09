"""
Backtesting engine for strategy simulation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from config.settings import get_settings
from data.fetcher import DataFetcher, get_data_fetcher
from data.indicators import TechnicalIndicators
from backtest.cost_model import CostModel, CostConfig
from backtest.metrics import PerformanceMetrics, MetricsConfig


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: date = None
    end_date: date = None
    initial_capital: float = 100000.0
    rebalance_freq: str = "monthly"
    commission_fixed: float = 0.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    risk_free_rate: float = 0.03
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = date.today() - timedelta(days=365*3)
        if self.end_date is None:
            self.end_date = date.today()


@dataclass
class Trade:
    """Represents a single trade."""
    date: date
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    value: float
    cost: float
    
    def to_dict(self) -> dict:
        return {
            'date': self.date.isoformat() if isinstance(self.date, date) else str(self.date),
            'ticker': self.ticker,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'value': self.value,
            'cost': self.cost,
        }


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    portfolio_values: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    drawdown_series: pd.Series
    weights_history: pd.DataFrame
    config: BacktestConfig
    success: bool = True
    message: str = ""
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


class BacktestEngine:
    """
    Main backtesting engine.
    Supports static weight and dynamic strategy backtests.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.data_fetcher = get_data_fetcher()
        
        # Initialize cost model
        cost_config = CostConfig(
            commission_fixed=self.config.commission_fixed,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
        )
        self.cost_model = CostModel(cost_config)
        
        # Initialize metrics calculator
        metrics_config = MetricsConfig(risk_free_rate=self.config.risk_free_rate)
        self.metrics = PerformanceMetrics(metrics_config)
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Get rebalancing dates based on frequency.
        
        Args:
            dates: All trading dates
            
        Returns:
            List of rebalancing dates
        """
        freq = self.config.rebalance_freq.lower()
        
        if freq == "daily":
            return list(dates)
        
        elif freq == "weekly":
            # Rebalance on Mondays (or first trading day of week)
            weekly = dates.to_series().groupby(pd.Grouper(freq='W')).first()
            return [d for d in weekly.dropna().values]
        
        elif freq == "monthly":
            # Rebalance on first trading day of month
            monthly = dates.to_series().groupby(pd.Grouper(freq='M')).first()
            return [d for d in monthly.dropna().values]
        
        elif freq == "quarterly":
            # Rebalance on first trading day of quarter
            quarterly = dates.to_series().groupby(pd.Grouper(freq='Q')).first()
            return [d for d in quarterly.dropna().values]
        
        else:
            # Default to monthly
            monthly = dates.to_series().groupby(pd.Grouper(freq='M')).first()
            return [d for d in monthly.dropna().values]
    
    def run_static(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        config: Optional[BacktestConfig] = None
    ) -> BacktestResult:
        """
        Run backtest with static (buy-and-hold) weights.
        
        Args:
            tickers: List of tickers
            weights: Dictionary of ticker -> weight (percentage)
            config: Optional custom config
            
        Returns:
            BacktestResult
        """
        cfg = config or self.config
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="Weights sum to zero",
            )
        
        norm_weights = {t: w / total_weight for t, w in weights.items()}
        
        # Fetch price data
        prices = self.data_fetcher.fetch_prices(
            tickers,
            start_date=cfg.start_date.strftime('%Y-%m-%d'),
            end_date=cfg.end_date.strftime('%Y-%m-%d'),
        )
        
        if prices.empty:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="No price data available",
            )
        
        # Filter to available tickers
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="No valid tickers found",
            )
        
        # Recalculate weights for available tickers
        avail_weights = {t: norm_weights.get(t, 0) for t in available}
        avail_total = sum(avail_weights.values())
        if avail_total > 0:
            avail_weights = {t: w / avail_total for t, w in avail_weights.items()}
        
        # Calculate normalized prices
        norm_prices = prices[available] / prices[available].iloc[0]
        
        # Calculate portfolio value
        portfolio_value = pd.Series(0.0, index=norm_prices.index)
        for ticker in available:
            weight = avail_weights.get(ticker, 0)
            portfolio_value += norm_prices[ticker] * (cfg.initial_capital * weight)
        
        # Calculate metrics
        metrics_dict = self.metrics.calculate_all(portfolio_value)
        
        # Calculate drawdown
        drawdown = self.metrics.drawdown_series(portfolio_value)
        
        # Create weights history (static)
        weights_history = pd.DataFrame(
            {t: [avail_weights.get(t, 0) * 100] * len(norm_prices) for t in available},
            index=norm_prices.index
        )
        
        # No trades in static backtest (buy and hold)
        trades = []
        
        return BacktestResult(
            portfolio_values=portfolio_value,
            trades=trades,
            metrics=metrics_dict,
            drawdown_series=drawdown,
            weights_history=weights_history,
            config=cfg,
            success=True,
        )
    
    def run_dynamic(
        self,
        tickers: List[str],
        initial_weights: Dict[str, float],
        strategy_func: Callable,
        config: Optional[BacktestConfig] = None
    ) -> BacktestResult:
        """
        Run backtest with dynamic strategy.
        
        Args:
            tickers: List of tickers
            initial_weights: Starting weights
            strategy_func: Function that takes (context, date) and returns target weights
            config: Optional custom config
            
        Returns:
            BacktestResult
        """
        cfg = config or self.config
        
        # Normalize initial weights
        total_weight = sum(initial_weights.values())
        if total_weight == 0:
            norm_weights = {t: 100 / len(tickers) for t in tickers}
        else:
            norm_weights = {t: (w / total_weight) * 100 for t, w in initial_weights.items()}
        
        # Fetch price data (with extra lookback for indicators)
        lookback_start = cfg.start_date - timedelta(days=300)
        
        prices = self.data_fetcher.fetch_prices(
            tickers,
            start_date=lookback_start.strftime('%Y-%m-%d'),
            end_date=cfg.end_date.strftime('%Y-%m-%d'),
        )
        
        if prices.empty:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="No price data available",
            )
        
        # Filter to backtest period
        backtest_prices = prices[prices.index >= pd.Timestamp(cfg.start_date)]
        
        if backtest_prices.empty:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="No data in backtest period",
            )
        
        # Filter to available tickers
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return BacktestResult(
                portfolio_values=pd.Series(dtype=float),
                trades=[],
                metrics={},
                drawdown_series=pd.Series(dtype=float),
                weights_history=pd.DataFrame(),
                config=cfg,
                success=False,
                message="No valid tickers found",
            )
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(backtest_prices.index)
        
        # Initialize tracking variables
        current_weights = {t: norm_weights.get(t, 0) for t in available}
        portfolio_values = []
        weights_history = []
        trades = []
        
        # Calculate initial position values
        cash = cfg.initial_capital
        positions = {t: 0.0 for t in available}  # Number of shares
        
        # Initial purchase
        first_date = backtest_prices.index[0]
        first_prices = backtest_prices.iloc[0]
        
        for ticker in available:
            weight = current_weights.get(ticker, 0) / 100
            if weight > 0:
                value_to_invest = cash * weight
                price = first_prices[ticker]
                shares = value_to_invest / price
                cost = self.cost_model.calculate_total_cost(value_to_invest, price)
                
                positions[ticker] = shares
                
                trades.append(Trade(
                    date=first_date.date() if hasattr(first_date, 'date') else first_date,
                    ticker=ticker,
                    action='BUY',
                    shares=shares,
                    price=price,
                    value=value_to_invest,
                    cost=cost,
                ))
        
        cash = 0  # All invested
        
        # Simulate through time
        for idx, row in backtest_prices.iterrows():
            current_date = idx.date() if hasattr(idx, 'date') else idx
            
            # Calculate current portfolio value
            position_values = {}
            total_value = cash
            
            for ticker in available:
                if ticker in row and positions.get(ticker, 0) > 0:
                    price = row[ticker]
                    value = positions[ticker] * price
                    position_values[ticker] = value
                    total_value += value
            
            portfolio_values.append({'date': idx, 'value': total_value})
            
            # Calculate current weights
            if total_value > 0:
                current_weights = {t: (position_values.get(t, 0) / total_value) * 100 
                                   for t in available}
            
            weights_history.append({
                'date': idx,
                **{t: current_weights.get(t, 0) for t in available}
            })
            
            # Check if rebalancing date
            if idx in rebalance_dates:
                try:
                    # Create strategy context
                    from strategy.engine import StrategyContext
                    
                    ctx = StrategyContext(
                        tickers=available,
                        current_weights=current_weights,
                        current_date=current_date,
                        data_fetcher=self.data_fetcher,
                        lookback_days=252,
                    )
                    
                    # Execute strategy
                    target_weights = strategy_func(ctx, current_date)
                    
                    if target_weights is None:
                        continue
                    
                    # Normalize target weights
                    target_total = sum(target_weights.values())
                    if target_total > 0:
                        target_weights = {t: (w / target_total) * 100 
                                         for t, w in target_weights.items()}
                    
                    # Execute rebalancing trades
                    for ticker in available:
                        current_w = current_weights.get(ticker, 0)
                        target_w = target_weights.get(ticker, 0)
                        
                        diff = target_w - current_w
                        
                        # Skip small changes
                        if abs(diff) < 1.0:  # Less than 1% change
                            continue
                        
                        price = row[ticker]
                        current_value = position_values.get(ticker, 0)
                        target_value = total_value * (target_w / 100)
                        trade_value = target_value - current_value
                        
                        if abs(trade_value) < 100:  # Skip tiny trades
                            continue
                        
                        # Calculate trade
                        shares_change = trade_value / price
                        cost = self.cost_model.calculate_total_cost(abs(trade_value), price)
                        
                        if trade_value > 0:  # Buy
                            positions[ticker] = positions.get(ticker, 0) + shares_change
                            action = 'BUY'
                        else:  # Sell
                            positions[ticker] = max(0, positions.get(ticker, 0) + shares_change)
                            action = 'SELL'
                        
                        trades.append(Trade(
                            date=current_date,
                            ticker=ticker,
                            action=action,
                            shares=abs(shares_change),
                            price=price,
                            value=abs(trade_value),
                            cost=cost,
                        ))
                        
                        # Deduct cost from portfolio
                        # (simplified - in reality would affect cash)
                    
                    current_weights = target_weights
                    
                except Exception as e:
                    # Strategy error - continue with current weights
                    pass
        
        # Convert to Series/DataFrame
        values_df = pd.DataFrame(portfolio_values)
        portfolio_series = pd.Series(
            values_df['value'].values,
            index=pd.DatetimeIndex(values_df['date'])
        )
        
        weights_df = pd.DataFrame(weights_history)
        if 'date' in weights_df.columns:
            weights_df.set_index('date', inplace=True)
        
        # Calculate metrics
        metrics_dict = self.metrics.calculate_all(
            portfolio_series,
            trades=[t.to_dict() for t in trades]
        )
        
        # Calculate total costs
        total_costs = sum(t.cost for t in trades)
        metrics_dict['Total Trading Costs ($)'] = total_costs
        metrics_dict['Trade Count'] = len(trades)
        
        # Calculate drawdown
        drawdown = self.metrics.drawdown_series(portfolio_series)
        
        return BacktestResult(
            portfolio_values=portfolio_series,
            trades=trades,
            metrics=metrics_dict,
            drawdown_series=drawdown,
            weights_history=weights_df,
            config=cfg,
            success=True,
        )
    
    def run_with_code(
        self,
        tickers: List[str],
        initial_weights: Dict[str, float],
        strategy_code: str,
        config: Optional[BacktestConfig] = None
    ) -> BacktestResult:
        """
        Run backtest with strategy code string.
        
        Args:
            tickers: List of tickers
            initial_weights: Starting weights
            strategy_code: Python strategy code
            config: Optional custom config
            
        Returns:
            BacktestResult
        """
        from strategy.engine import StrategyEngine
        
        engine = StrategyEngine()
        
        def strategy_func(ctx, current_date):
            result = engine.execute(
                code=strategy_code,
                tickers=ctx.tickers,
                current_weights=ctx.get_current_weights(),
                current_date=current_date,
            )
            
            if result.success:
                return result.target_weights
            return None
        
        return self.run_dynamic(
            tickers=tickers,
            initial_weights=initial_weights,
            strategy_func=strategy_func,
            config=config,
        )
