import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Local imports
from backtesting_engine.trade_constructor import StoreTrades, Trade
from backtesting_engine.backtesting_engine import BacktestEngine
from risk_management.position_sizing import PositionSizer
from risk_management.drawdown_limit import DrawdownLimiter
from backtesting_engine.run_backtest import main as run_main

def main():
    """
    Test version of the backtest that uses a smaller dataset
    and different risk parameters for quick testing.
    """
    # Override environment variables to set test parameters
    os.environ['MAX_TRADES'] = '100'  # Smaller dataset for testing
    os.environ['RISK_PER_TRADE'] = '0.02'  # Higher risk for testing
    os.environ['MAX_DRAWDOWN'] = '0.15'  # Higher drawdown limit for testing
    os.environ['OUTPUT_PREFIX'] = 'test_'  # Prefix for output files
    
    # Call the main backtest function with test settings
    run_main()

if __name__ == "__main__":
    main() 