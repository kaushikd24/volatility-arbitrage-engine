import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Any, Callable

# Add root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Local imports when needed
from backtesting_engine.trade_constructor import StoreTrades, Trade
from backtesting_engine.backtesting_engine import BacktestEngine
from risk_management.position_sizing import PositionSizer
from risk_management.drawdown_limit import DrawdownLimiter

def parameter_sweep(
    backtest_fn: Callable,
    param_grid: Dict[str, List[Any]],
    fixed_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Run a parameter sweep over multiple risk management parameters.
    
    Parameters:
    -----------
    backtest_fn : callable
        Function that runs a backtest and returns performance metrics
    param_grid : dict
        Dictionary mapping parameter names to lists of values to test
    fixed_params : dict, optional
        Dictionary of parameters that remain constant across all runs
        
    Returns:
    --------
    pd.DataFrame
        Results of parameter sweep with columns for parameters and performance metrics
    """
    if fixed_params is None:
        fixed_params = {}
    
    print(f"Running parameter sweep with {len(param_grid)} parameters")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    
    results = []
    
    # Run backtest for each parameter combination
    for i, combo in enumerate(param_combinations):
        # Create parameter dictionary for this run
        run_params = fixed_params.copy()
        for name, value in zip(param_names, combo):
            run_params[name] = value
        
        # Print progress
        param_str = ", ".join([f"{name}={value}" for name, value in zip(param_names, combo)])
        print(f"Running combination {i+1}/{len(param_combinations)}: {param_str}")
        
        # Run backtest with these parameters
        metrics = backtest_fn(**run_params)
        
        # Store results
        result = {**run_params, **metrics}
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def run_backtest_with_params(
    risk_per_trade: float, 
    max_drawdown_pct: float,
    initial_capital: float = 100000.0,
    max_trades: int = 100
) -> Dict[str, Any]:
    """Run a backtest with specific risk management parameters."""
    
    print(f"Running backtest with: risk_per_trade={risk_per_trade}, max_drawdown_pct={max_drawdown_pct}")
    
    # Define paths relative to project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    signals_path = os.path.join(root_dir, 'data', 'signals_mispricing.csv')
    market_data_path = os.path.join(root_dir, 'data', 'clean', 'SPY_2023_eod.csv')
    
    # Load data
    signals = pd.read_csv(signals_path)
    market_data = pd.read_csv(market_data_path)
    
    # Take a small sample for faster parameter sweep
    if len(signals) > max_trades:
        signals = signals.sample(max_trades, random_state=42)
    
    # Merge data
    keys = ['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE']
    merged_df = pd.merge(signals, market_data, on=keys, how='inner')
    
    if merged_df.empty:
        return {
            "win_rate": 0,
            "total_pnl": 0,
            "profit_factor": 0,
            "final_equity": initial_capital,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }
    
    # Initialize risk management components
    position_sizer = PositionSizer(capital=initial_capital, risk_per_trade=risk_per_trade)
    drawdown_limiter = DrawdownLimiter(starting_equity=initial_capital, max_drawdown_pct=max_drawdown_pct)
    
    # Create trade data
    trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'action', 'position_type', 'confidence']]
    trade_data['entry_price'] = merged_df[['P_BID', 'P_ASK']].mean(axis=1).values
    
    # Create trades with position sizing
    trades = StoreTrades()
    for _, row in trade_data.iterrows():
        quantity = position_sizer.size_position(
            entry_price=row['entry_price'], 
            confidence=row.get('confidence', None)
        )
        
        trade = Trade(
            entry_date=row['QUOTE_DATE'],
            exit_date=row['EXPIRE_DATE'],
            strike=row['STRIKE'],
            action=row['action'],
            position_type=row['position_type'],
            entry_price=row['entry_price'],
            confidence=row['confidence'],
            quantity=quantity
        )
        trades.add_trade(trade)
    
    # Run backtest
    backtest = BacktestEngine(trades, trade_data, market_data)
    results = backtest.run()
    
    if results.empty:
        return {
            "win_rate": 0,
            "total_pnl": 0,
            "profit_factor": 0,
            "final_equity": initial_capital,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }
    
    # Apply drawdown limit
    current_equity = initial_capital
    equity_curve = [initial_capital]
    trading_should_continue = True
    
    # Process results with drawdown limit
    for idx, row in results.iterrows():
        if not trading_should_continue:
            continue
            
        current_equity += row['pnl']
        equity_curve.append(current_equity)
        
        trading_should_continue = drawdown_limiter.update_equity(current_equity)
    
    # Calculate metrics
    metrics = backtest.calculate_performance_metrics(results)
    
    # Calculate drawdown and Sharpe ratio
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate returns for Sharpe ratio
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        "win_rate": metrics['win_rate'],
        "total_pnl": metrics['total_pnl'],
        "avg_pnl": metrics['avg_pnl'],
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": metrics['profit_factor'],
        "final_equity": current_equity
    }

def run_quick_parameter_sweep():
    """
    Run a quick parameter sweep with a small set of parameters.
    """
    # Define a small set of parameters
    param_grid = {
        "risk_per_trade": [0.01, 0.02, 0.03],
        "max_drawdown_pct": [0.1, 0.15, 0.2],
    }
    
    # Fixed parameters
    fixed_params = {
        "initial_capital": 100000.0,
        "max_trades": 100  # Small sample for quick testing
    }
    
    # Run parameter sweep
    print("Starting quick parameter sweep...")
    results = parameter_sweep(run_backtest_with_params, param_grid, fixed_params)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'parameter_sweep_results.csv')
    results.to_csv(results_path, index=False)
    
    # Display results
    print("\n=== PARAMETER SWEEP RESULTS ===")
    print(results)
    
    # Sort by different metrics
    best_equity = results.sort_values("final_equity", ascending=False).head(3)
    print("\nBest parameter sets by final equity:")
    print(best_equity[['risk_per_trade', 'max_drawdown_pct', 'final_equity', 'win_rate']])
    
    best_sharpe = results.sort_values("sharpe_ratio", ascending=False).head(3)
    if 'sharpe_ratio' in best_sharpe.columns:
        print("\nBest parameter sets by Sharpe ratio:")
        print(best_sharpe[['risk_per_trade', 'max_drawdown_pct', 'sharpe_ratio', 'final_equity']])
    
    # Create a simple bar chart
    plt.figure(figsize=(12, 8))
    risk_values = sorted(param_grid["risk_per_trade"])
    
    for i, risk in enumerate(risk_values):
        subset = results[results["risk_per_trade"] == risk]
        x_labels = [f"{drawdown}%" for drawdown in sorted(param_grid["max_drawdown_pct"])]
        x_positions = np.arange(len(x_labels)) + (i - len(risk_values)/2 + 0.5) * 0.25
        
        plt.bar(
            x_positions,
            subset.sort_values("max_drawdown_pct")["final_equity"].values,
            width=0.2,
            label=f"Risk {risk:.1%}"
        )
    
    plt.title('Final Equity by Risk Parameters')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Final Equity ($)')
    plt.legend()
    plt.xticks(range(len(x_labels)), x_labels)
    plt.tight_layout()
    
    chart_path = os.path.join(results_dir, 'parameter_comparison.png')
    plt.savefig(chart_path)
    
    print(f"\nResults saved to {results_path}")
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    run_quick_parameter_sweep() 