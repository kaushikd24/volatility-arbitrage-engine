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
    
    # Take a small sample for testing
    sample_size = 20
    print(f"Taking a sample of {sample_size} signals for testing")
    signals_sample = signals.sample(sample_size, random_state=42)
    
    # Merge data and prepare trade data
    keys = ['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE']
    merged_df = pd.merge(signals_sample, market_data, on=keys, how='inner')
    
    print(f"Merged data shape: {merged_df.shape}")
    
    if merged_df.empty:
        print("ERROR: No matching data found after merging signals and market data")
        # Try a different approach
        print("Trying a more flexible merge...")
        
        # Convert date columns to datetime
        for df in [signals_sample, market_data]:
            for col in ['QUOTE_DATE', 'EXPIRE_DATE']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Merge only on STRIKE first
        merged_df = pd.merge(signals_sample, market_data, on=['STRIKE'], how='inner')
        print(f"Merged on STRIKE only: {merged_df.shape}")
        
        if merged_df.empty:
            print("Still no matches. Checking data samples:")
            print("\nSignals sample:")
            print(signals_sample[keys].head())
            print("\nMarket data sample:")
            print(market_data[keys].head())
            return
    
    # Print sample trade data
    print("\nColumns in merged_df:")
    print(merged_df.columns.tolist())
    
    # Create trade data - use the actual column names from the merged DataFrame
    trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'action', 'position_type', 'confidence']]
    
    # Check if P_BID and P_ASK columns exist
    if 'P_BID' in merged_df.columns and 'P_ASK' in merged_df.columns:
        trade_data['entry_price'] = merged_df[['P_BID', 'P_ASK']].mean(axis=1)
    else:
        print("WARNING: P_BID or P_ASK columns missing. Using placeholder entry price of 1.0")
        trade_data['entry_price'] = 1.0
    
    # Print sample trade data
    print("\nSample trade data:")
    print(trade_data.head())
    
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
    results_csv_path = os.path.join(results_dir, 'test_backtest_results.csv')
    results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    
    # Print sample results
    print("\nSample results:")
    print(results.head())

if __name__ == "__main__":
    main() 