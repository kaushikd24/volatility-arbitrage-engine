import pandas as pd
import numpy as np
from .trade_constructor import Trade, StoreTrades
from datetime import datetime, timedelta

class BacktestEngine:
    def __init__(self, trades, trade_data: pd.DataFrame, market_data: pd.DataFrame):
        self.trades = trades.get_trades() if hasattr(trades, 'get_trades') else trades
        self.trade_data = trade_data
        self.market_data = market_data
        
        # Convert dates to datetime for proper comparison
        for df in [self.market_data, self.trade_data]:
            for col in ['QUOTE_DATE', 'EXPIRE_DATE']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Print date range of market data for reference
        print(f"Market data date range: {self.market_data['QUOTE_DATE'].min().date()} to {self.market_data['QUOTE_DATE'].max().date()}")
        
    def run(self):
        results = []
        processed_count = 0
        skipped_count = 0
        data_range_issue_count = 0
        
        print("Checking trades against available market data...")
        earliest_data = self.market_data['QUOTE_DATE'].min().date()
        latest_data = self.market_data['QUOTE_DATE'].max().date()
        
        # Filter trades to only include those within our data range
        valid_trades = []
        for trade in self.trades:
            entry_date = pd.to_datetime(trade.entry_date).date() if isinstance(trade.entry_date, str) else trade.entry_date.date()
            exit_date = pd.to_datetime(trade.exit_date).date() if isinstance(trade.exit_date, str) else trade.exit_date.date()
            
            # Check if trade dates fall within our data range
            if entry_date < earliest_data or exit_date > latest_data:
                data_range_issue_count += 1
                continue
                
            valid_trades.append(trade)
            
        print(f"Found {len(valid_trades)} trades within market data date range.")
        print(f"Skipped {data_range_issue_count} trades due to dates outside market data range.")
        
        for trade in valid_trades:
            try:
                # Get entry and exit dates
                entry_date = pd.to_datetime(trade.entry_date) if isinstance(trade.entry_date, str) else trade.entry_date
                exit_date = pd.to_datetime(trade.exit_date) if isinstance(trade.exit_date, str) else trade.exit_date
                
                # Find the underlying price at exit for this specific strike
                exit_mask = (
                    (self.market_data['QUOTE_DATE'] == exit_date) & 
                    (self.market_data['STRIKE'] == trade.strike)
                )
                
                exit_data = self.market_data[exit_mask]
                
                if exit_data.empty:
                    print(f"Warning: No exact match for exit date {exit_date.date()} with strike {trade.strike}")
                    # Try to find close dates (within 1 day)
                    alternative_mask = (
                        (self.market_data['QUOTE_DATE'] >= exit_date - timedelta(days=1)) &
                        (self.market_data['QUOTE_DATE'] <= exit_date + timedelta(days=1)) &
                        (self.market_data['STRIKE'] == trade.strike)
                    )
                    alternative_data = self.market_data[alternative_mask]
                    
                    if alternative_data.empty:
                        print(f"  Skipping trade - no suitable exit data found")
                        skipped_count += 1
                        continue
                    
                    # Take the closest date
                    alternative_data['date_diff'] = abs(alternative_data['QUOTE_DATE'] - exit_date)
                    alternative_data = alternative_data.sort_values('date_diff')
                    exit_data = alternative_data.iloc[[0]]
                    print(f"  Using {exit_data['QUOTE_DATE'].iloc[0].date()} instead (closest available)")
                
                # Get underlying price
                underlying_price = exit_data['UNDERLYING_LAST'].iloc[0]
                
                # Calculate exit price using the Trade class method for put options
                exit_price = trade.calc_exit_price(underlying_price)
                
                # Calculate profit/loss
                if trade.action == 'buy':
                    pnl = exit_price - trade.entry_price
                else:  # 'sell'
                    pnl = trade.entry_price - exit_price
                
                # Store result
                result = {
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'actual_exit_date': exit_data['QUOTE_DATE'].iloc[0],
                    'strike': trade.strike,
                    'action': trade.action,
                    'position_type': trade.position_type,
                    'entry_price': trade.entry_price,
                    'exit_price': exit_price,
                    'underlying_price': underlying_price,
                    'pnl': pnl,
                    'pnl_percent': (pnl / trade.entry_price) * 100 if trade.entry_price != 0 else 0,
                    'confidence': trade.confidence
                }
                
                results.append(result)
                processed_count += 1
                
                # Print progress every 100 trades
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} trades...")
                
            except Exception as e:
                print(f"Error processing trade: {e}")
                skipped_count += 1
                continue
        
        print(f"Successfully processed {processed_count} trades.")
        print(f"Skipped {skipped_count} trades due to missing data or errors.")
        
        # Convert results to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            # Sort by entry date
            results_df = results_df.sort_values('entry_date')
            return results_df
        else:
            return pd.DataFrame()
            
    def calculate_performance_metrics(self, results_df):
        """
        Calculate performance metrics based on backtest results
        """
        if results_df.empty:
            return {}
            
        metrics = {
            'total_trades': len(results_df),
            'winning_trades': len(results_df[results_df['pnl'] > 0]),
            'losing_trades': len(results_df[results_df['pnl'] < 0]),
            'win_rate': len(results_df[results_df['pnl'] > 0]) / len(results_df) if len(results_df) > 0 else 0,
            'total_pnl': results_df['pnl'].sum(),
            'avg_pnl': results_df['pnl'].mean(),
            'max_profit': results_df['pnl'].max(),
            'max_loss': results_df['pnl'].min(),
            'avg_win': results_df[results_df['pnl'] > 0]['pnl'].mean() if len(results_df[results_df['pnl'] > 0]) > 0 else 0,
            'avg_loss': results_df[results_df['pnl'] < 0]['pnl'].mean() if len(results_df[results_df['pnl'] < 0]) > 0 else 0,
            'profit_factor': abs(results_df[results_df['pnl'] > 0]['pnl'].sum() / results_df[results_df['pnl'] < 0]['pnl'].sum()) 
                             if results_df[results_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
        }
        
        return metrics 