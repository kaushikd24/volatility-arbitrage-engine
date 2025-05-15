"""
Volatility Arbitrage Backtesting Engine

This package contains tools for backtesting volatility arbitrage strategies:
- Trade construction and management
- Backtesting engine
- Performance analysis
"""

from .trade_constructor import Trade, StoreTrades
from .backtesting_engine import BacktestEngine

__all__ = ['Trade', 'StoreTrades', 'BacktestEngine'] 