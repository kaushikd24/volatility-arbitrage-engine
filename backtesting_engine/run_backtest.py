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

def main():
    print("Loading data...")
    # Define paths relative to project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    signals_path = os.path.join(root_dir, 'data', 'signals_mispricing.csv')
    market_data_path = os.path.join(root_dir, 'data', 'clean', 'SPY_2023_eod.csv')
    results_dir = os.path.join(root_dir, 'results')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the data
    signals = pd.read_csv(signals_path)
    market_data = pd.read_csv(market_data_path)
    
    # Print some basic info about the data
    print(f"Signals shape: {signals.shape}")
    print(f"Market data shape: {market_data.shape}")
    
    # Set maximum trades to process to avoid overwhelming the system
    max_trades = 1000  # Adjust based on performance
    if len(signals) > max_trades:
        print(f"Limiting analysis to {max_trades} signals")
        signals = signals.sample(max_trades, random_state=42)
    
    # Merge data and prepare trade data
    keys = ['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE']
    merged_df = pd.merge(signals, market_data, on=keys, how='inner')
    
    print(f"Merged data shape: {merged_df.shape}")
    
    if merged_df.empty:
        print("ERROR: No matching data found after merging signals and market data")
        return
    
    # Create trade data
    trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'action', 'position_type', 'confidence']]
    trade_data['entry_price'] = merged_df[['P_BID', 'P_ASK']].mean(axis=1)
    
    # Create trades
    trades = StoreTrades()
    for _, row in trade_data.iterrows():
        trade = Trade(
            entry_date=row['QUOTE_DATE'],
            exit_date=row['EXPIRE_DATE'],
            strike=row['STRIKE'],
            action=row['action'],
            position_type=row['position_type'],
            entry_price=row['entry_price'],
            confidence=row['confidence'],
            quantity=1
        )
        trades.add_trade(trade)
    
    print(f"Created {len(trades.get_trades())} trades")
    
    # Initialize and run backtesting engine
    print("Running backtest...")
    backtest = BacktestEngine(trades, trade_data, market_data)
    results = backtest.run()
    
    # Check if we got results
    if results.empty:
        print("No results generated. Check for errors.")
        return
    
    # Calculate and display performance metrics
    metrics = backtest.calculate_performance_metrics(results)
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Average P&L: ${metrics['avg_pnl']:.2f}")
    print(f"Max Profit: ${metrics['max_profit']:.2f}")
    print(f"Max Loss: ${metrics['max_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Save results to CSV
    results_csv_path = os.path.join(results_dir, 'backtest_results.csv')
    results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    
    # Create a simple plot of cumulative P&L
    plt.figure(figsize=(10, 6))
    cumulative_pnl = results['pnl'].cumsum()
    plt.plot(cumulative_pnl.index, cumulative_pnl.values)
    plt.title('Cumulative P&L')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P&L ($)')
    plt.grid(True)
    pnl_plot_path = os.path.join(results_dir, 'cumulative_pnl.png')
    plt.savefig(pnl_plot_path)
    plt.close()
    
    # Show P&L by confidence level
    if 'confidence' in results.columns:
        plt.figure(figsize=(10, 6))
        results.boxplot(column='pnl', by='confidence')
        plt.title('P&L by Confidence Level')
        plt.suptitle('')
        plt.ylabel('P&L ($)')
        confidence_plot_path = os.path.join(results_dir, 'pnl_by_confidence.png')
        plt.savefig(confidence_plot_path)
        plt.close()
    
    print("Plots saved to results/ directory")

if __name__ == "__main__":
    main() 