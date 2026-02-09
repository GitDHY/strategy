"""
Transaction cost model for backtesting.
Handles commissions, slippage, and other trading costs.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class CostConfig:
    """Configuration for transaction costs."""
    commission_fixed: float = 0.0      # Fixed cost per trade (e.g., $1)
    commission_pct: float = 0.001      # Percentage commission (e.g., 0.1%)
    slippage_pct: float = 0.001        # Slippage as percentage (e.g., 0.1%)
    min_trade_value: float = 100.0     # Minimum trade value to execute
    
    def __post_init__(self):
        """Validate cost configuration."""
        assert self.commission_fixed >= 0, "Fixed commission must be non-negative"
        assert 0 <= self.commission_pct <= 0.1, "Commission percentage must be between 0 and 10%"
        assert 0 <= self.slippage_pct <= 0.1, "Slippage percentage must be between 0 and 10%"


class CostModel:
    """
    Transaction cost model for realistic backtesting.
    Calculates commissions, slippage, and market impact.
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost model.
        
        Args:
            config: Cost configuration
        """
        self.config = config or CostConfig()
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            trade_value: Absolute value of the trade
            
        Returns:
            Commission cost
        """
        if trade_value < self.config.min_trade_value:
            return 0.0
        
        fixed = self.config.commission_fixed
        variable = abs(trade_value) * self.config.commission_pct
        
        return fixed + variable
    
    def calculate_slippage(
        self,
        trade_value: float,
        price: float,
        volatility: float = None,
        is_buy: bool = True
    ) -> float:
        """
        Calculate slippage cost for a trade.
        
        Args:
            trade_value: Value of the trade
            price: Current price
            volatility: Asset volatility (optional, for dynamic slippage)
            is_buy: True if buying, False if selling
            
        Returns:
            Slippage cost (always positive)
        """
        if abs(trade_value) < self.config.min_trade_value:
            return 0.0
        
        # Base slippage
        base_slippage = abs(trade_value) * self.config.slippage_pct
        
        # Volatility adjustment (higher volatility = more slippage)
        if volatility is not None and volatility > 0:
            vol_multiplier = min(2.0, 1 + volatility / 0.2)  # Cap at 2x
            base_slippage *= vol_multiplier
        
        return base_slippage
    
    def calculate_total_cost(
        self,
        trade_value: float,
        price: float = None,
        volatility: float = None,
        is_buy: bool = True
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            trade_value: Value of the trade
            price: Current price
            volatility: Asset volatility
            is_buy: True if buying
            
        Returns:
            Total cost (commission + slippage)
        """
        commission = self.calculate_commission(trade_value)
        slippage = self.calculate_slippage(trade_value, price, volatility, is_buy)
        
        return commission + slippage
    
    def calculate_rebalance_cost(
        self,
        current_values: Dict[str, float],
        target_values: Dict[str, float],
        prices: Dict[str, float] = None,
        volatilities: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate total cost for a rebalancing operation.
        
        Args:
            current_values: Current position values
            target_values: Target position values
            prices: Current prices (optional)
            volatilities: Asset volatilities (optional)
            
        Returns:
            Dictionary with breakdown:
            - 'total_cost': Total rebalancing cost
            - 'commission': Total commissions
            - 'slippage': Total slippage
            - 'turnover': Total turnover value
            - 'by_asset': Per-asset cost breakdown
        """
        total_commission = 0.0
        total_slippage = 0.0
        total_turnover = 0.0
        by_asset = {}
        
        all_tickers = set(current_values.keys()) | set(target_values.keys())
        
        for ticker in all_tickers:
            current = current_values.get(ticker, 0)
            target = target_values.get(ticker, 0)
            trade_value = target - current
            
            if abs(trade_value) < self.config.min_trade_value:
                continue
            
            price = prices.get(ticker, 1.0) if prices else 1.0
            vol = volatilities.get(ticker) if volatilities else None
            is_buy = trade_value > 0
            
            commission = self.calculate_commission(abs(trade_value))
            slippage = self.calculate_slippage(abs(trade_value), price, vol, is_buy)
            
            total_commission += commission
            total_slippage += slippage
            total_turnover += abs(trade_value)
            
            by_asset[ticker] = {
                'trade_value': trade_value,
                'commission': commission,
                'slippage': slippage,
                'total_cost': commission + slippage,
            }
        
        return {
            'total_cost': total_commission + total_slippage,
            'commission': total_commission,
            'slippage': total_slippage,
            'turnover': total_turnover,
            'by_asset': by_asset,
        }
    
    def get_execution_price(
        self,
        price: float,
        is_buy: bool,
        volatility: float = None
    ) -> float:
        """
        Get effective execution price including slippage.
        
        Args:
            price: Market price
            is_buy: True if buying
            volatility: Asset volatility
            
        Returns:
            Execution price (higher for buys, lower for sells)
        """
        slippage_pct = self.config.slippage_pct
        
        # Adjust for volatility
        if volatility is not None and volatility > 0:
            vol_multiplier = min(2.0, 1 + volatility / 0.2)
            slippage_pct *= vol_multiplier
        
        if is_buy:
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)
