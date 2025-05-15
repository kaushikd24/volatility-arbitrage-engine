import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime as dt

# Add root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Local imports
from backtesting_engine.trade_constructor import StoreTrades, Trade
from backtesting_engine.backtesting_engine import BacktestEngine
from risk_management.position_sizing import PositionSizer
from risk_management.drawdown_limit import DrawdownLimiter

def main():
    """
    Main function to run the backtest with risk management.
    Can be configured through environment variables:
    - MAX_TRADES: Maximum number of trades to process (default: 1000)
    - RISK_PER_TRADE: Fraction of capital to risk per trade (default: 0.01)
    - MAX_DRAWDOWN: Maximum drawdown limit as decimal (default: 0.1)
    - OUTPUT_PREFIX: Prefix for output files (default: "")
    - MAX_POSITION_PCT: Maximum position size as percentage of capital (default: 0.05)
    - ABSOLUTE_MAX_CONTRACTS: Maximum number of contracts per trade (default: 100)
    """
    print("Loading data:")
    
    # Get configuration from environment variables
    max_trades = int(os.environ.get('MAX_TRADES', 1000))
    risk_per_trade = float(os.environ.get('RISK_PER_TRADE', 0.01))
    max_drawdown = float(os.environ.get('MAX_DRAWDOWN', 0.1))
    output_prefix = os.environ.get('OUTPUT_PREFIX', '')
    max_position_pct = float(os.environ.get('MAX_POSITION_PCT', 0.05))
    absolute_max_contracts = int(os.environ.get('ABSOLUTE_MAX_CONTRACTS', 100))
    
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
    
    # Initialize risk management components
    initial_capital = 100000.0
    position_sizer = PositionSizer(
        capital=initial_capital, 
        risk_per_trade=risk_per_trade,
        max_position_pct=max_position_pct,
        absolute_max_contracts=absolute_max_contracts
    )
    drawdown_limiter = DrawdownLimiter(starting_equity=initial_capital, max_drawdown_pct=max_drawdown)
    
    # Create trade data
    trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'action', 'position_type', 'confidence']]
    
    # Ensure entry prices are reasonable
    # Calculate average of bid and ask, with a minimum value for sanity
    MIN_ENTRY_PRICE = 0.05  # Minimum $0.05 per contract
    
    # Use bid-ask midpoint for entry price, with a minimum value
    trade_data['entry_price'] = merged_df.apply(
        lambda row: max(
            (row['P_BID'] + row['P_ASK']) / 2 if pd.notna(row['P_BID']) and pd.notna(row['P_ASK']) else MIN_ENTRY_PRICE, 
            MIN_ENTRY_PRICE
        ), 
        axis=1
    )
    
    # Create trades with position sizing
    trades = StoreTrades()
    skipped_for_low_price = 0
    
    for _, row in trade_data.iterrows():
        # Skip trades with extremely low prices - likely not realistic
        if row['entry_price'] < MIN_ENTRY_PRICE:
            skipped_for_low_price += 1
            continue
            
        # Use position sizer to determine quantity
        quantity = position_sizer.size_position(
            entry_price=row['entry_price'],
            confidence=row.get('confidence', None)
        )
        
        # Skip trades where position sizer returned 0 quantity
        if quantity <= 0:
            continue
            
        trade = Trade(
            entry_date=row['QUOTE_DATE'],
            exit_date=row['EXPIRE_DATE'],
            strike=row['STRIKE'],
            action=row['action'],
            position_type=row['position_type'],
            entry_price=row['entry_price'],
            confidence=row['confidence'],
            quantity=quantity  # Use the quantity from position sizer
        )
        trades.add_trade(trade)
    
    print(f"Created {len(trades.get_trades())} trades")
    if skipped_for_low_price > 0:
        print(f"Skipped {skipped_for_low_price} trades due to unrealistically low prices")
    
    # Initialize and run backtesting engine
    print("Running backtest...")
    backtest = BacktestEngine(trades, trade_data, market_data)
    results = backtest.run()
    
    # Check if we got results
    if results.empty:
        print("No results generated. Check for errors.")
        return
    
    # Apply drawdown limit check
    current_equity = initial_capital
    trading_should_continue = True
    
    # Add equity curve to results
    results['equity'] = 0.0
    
    # Add maximum loss cap to avoid outliers
    max_loss_per_trade = initial_capital * 0.1  # Cap losses at 10% of capital per trade
    
    for idx, row in results.iterrows():
        if not trading_should_continue:
            # Mark trades after drawdown limit as 'skipped'
            results.loc[idx, 'status'] = 'skipped'
            continue
        
        # Apply maximum loss cap
        pnl = row['pnl']
        if pnl < -max_loss_per_trade:
            print(f"Capping large loss: ${pnl:.2f} limited to ${-max_loss_per_trade:.2f}")
            pnl = -max_loss_per_trade
            results.loc[idx, 'pnl'] = pnl
            
        current_equity += pnl
        results.loc[idx, 'equity'] = current_equity
        
        # Check if we should continue trading based on drawdown limit
        trading_should_continue = drawdown_limiter.update_equity(current_equity)
    
    # Calculate and display performance metrics
    metrics = backtest.calculate_performance_metrics(results)
    
    # Calculate CAGR
    if not results.empty:
        start_date = results['entry_date'].min()
        end_date = results['exit_date'].max()
        equity_series = results['equity']
        cagr = backtest.calculate_cagr(equity_series, start_date, end_date, initial_capital)
    else:
        cagr = 0.0
    
    print("\n=== BACKTEST RESULTS WITH RISK MANAGEMENT ===")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Risk Per Trade: {position_sizer.risk_per_trade:.2%}")
    print(f"Max Position Size: {position_sizer.max_position_pct:.2%}")
    print(f"Max Contracts: {position_sizer.absolute_max_contracts}")
    print(f"Max Drawdown Limit: {drawdown_limiter.max_drawdown_pct:.2%}")
    print(f"Final Equity: ${current_equity:.2f}")
    print(f"Total Return: {(current_equity/initial_capital - 1):.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Average P&L: ${metrics['avg_pnl']:.2f}")
    print(f"Max Profit: ${metrics['max_profit']:.2f}")
    print(f"Max Loss: ${metrics['max_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Check if trading was stopped due to drawdown limit
    if not trading_should_continue:
        print("\nWARNING: Trading was stopped due to exceeding maximum drawdown limit")
        skipped_trades = results[results['status'] == 'skipped'].shape[0]
        print(f"Number of trades skipped: {skipped_trades}")
    
    # Save results to CSV
    results_csv_path = os.path.join(results_dir, f'{output_prefix}backtest_results.csv')
    results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    
    # Create a simple plot of equity curve
    plt.figure(figsize=(12, 7))
    plt.plot(results['equity'])
    plt.title('Equity Curve with Risk Management')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    equity_plot_path = os.path.join(results_dir, f'{output_prefix}equity_curve.png')
    plt.savefig(equity_plot_path)
    plt.close()
    
    # Plot cumulative P&L
    plt.figure(figsize=(12, 7))
    cumulative_pnl = results['pnl'].cumsum()
    plt.plot(cumulative_pnl.index, cumulative_pnl.values)
    plt.title('Cumulative P&L')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P&L ($)')
    plt.grid(True)
    pnl_plot_path = os.path.join(results_dir, f'{output_prefix}cumulative_pnl.png')
    plt.savefig(pnl_plot_path)
    plt.close()
    
    # Show P&L by confidence level
    if 'confidence' in results.columns:
        plt.figure(figsize=(10, 6))
        results.boxplot(column='pnl', by='confidence')
        plt.title('P&L by Confidence Level')
        plt.suptitle('')
        plt.ylabel('P&L ($)')
        confidence_plot_path = os.path.join(results_dir, f'{output_prefix}pnl_by_confidence.png')
        plt.savefig(confidence_plot_path)
        plt.close()
    
    # Add performance summary with CAGR
    summary_data = {
        'Metric': ['Initial Capital', 'Final Equity', 'Total Return', 'CAGR', 
                  'Win Rate', 'Total Trades', 'Profit Factor', 'Max Drawdown'],
        'Value': [
            f"${initial_capital:.2f}",
            f"${current_equity:.2f}",
            f"{(current_equity/initial_capital - 1):.2%}",
            f"{cagr:.2%}",
            f"{metrics['win_rate']:.2%}",
            f"{metrics['total_trades']}",
            f"{metrics['profit_factor']:.2f}",
            f"{drawdown_limiter.max_drawdown_pct:.2%}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, f'{output_prefix}performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("Plots saved to results/ directory")
    print(f"Performance summary saved to {summary_path}")

if __name__ == "__main__":
    main() 