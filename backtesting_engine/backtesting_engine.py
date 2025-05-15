import pandas as pd
import numpy as np
import datetime as dt
from .trade_constructor import Trade, StoreTrades
from typing import Dict, Any, List, Tuple

class BacktestEngine:
    def __init__(self, trades: StoreTrades, trade_data: pd.DataFrame, market_data: pd.DataFrame):
        """
        Initialize the backtesting engine.
        
        Parameters:
        -----------
        trades : StoreTrades
            Object containing trades to backtest
        trade_data : pandas.DataFrame
            DataFrame with trade data
        market_data : pandas.DataFrame
            Market data for price lookup
        """
        self.trades = trades
        self.trade_data = trade_data
        self.market_data = market_data
        
        # Convert date columns to datetime
        for df in [self.trade_data, self.market_data]:
            for col in ['QUOTE_DATE', 'EXPIRE_DATE']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Print date range of market data for reference
        print(f"Market data date range: {self.market_data['QUOTE_DATE'].min().date()} to {self.market_data['QUOTE_DATE'].max().date()}")
        
    def run(self) -> pd.DataFrame:
        """
        Run the backtest by simulating the trades.
        
        """
        results = []
        
        # Get date range of market data
        market_dates = sorted(self.market_data['QUOTE_DATE'].unique())
        start_date = market_dates[0]
        end_date = market_dates[-1]
        print(f"Market data date range: {start_date.date()} to {end_date.date()}")
        
        # Filter trades to only include those within the market data date range
        all_trades = self.trades.get_trades()
        valid_trades = []
        skipped_trades = 0
        
        for trade in all_trades:
            if start_date <= trade.entry_date <= end_date and start_date <= trade.exit_date <= end_date:
                valid_trades.append(trade)
            else:
                skipped_trades += 1
        
        print("Checking trades against available market data...")
        print(f"Found {len(valid_trades)} trades within market data date range.")
        print(f"Skipped {skipped_trades} trades due to dates outside market data range.")
        
        # Process valid trades
        processed_count = 0
        error_count = 0
        
        for i, trade in enumerate(valid_trades):
            if i > 0 and i % 100 == 0:
                print(f"Processed {i} trades...")
            
            try:
                # Find exit price
                exit_price = self._find_exit_price(trade)
                if exit_price is None:
                    error_count += 1
                    continue
                
                # Calculate P&L
                pnl = self._calculate_trade_pnl(trade, exit_price)
                
                # Store result
                result = {
                    'trade_id': i,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'strike': trade.strike,
                    'action': trade.action,
                    'position_type': trade.position_type,
                    'entry_price': trade.entry_price,
                    'exit_price': exit_price,
                    'quantity': trade.quantity,
                    'pnl': pnl,
                    'status': 'executed',
                    'confidence': trade.confidence
                }
                results.append(result)
                processed_count += 1
            except Exception as e:
                print(f"Error processing trade {i}: {e}")
                error_count += 1
        
        print(f"Successfully processed {processed_count} trades.")
        print(f"Skipped {error_count} trades due to missing data or errors.")
        
        # Convert results to DataFrame
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Sort by date
        results_df = results_df.sort_values('entry_date')
        
        return results_df
    
    def _find_exit_price(self, trade: Trade) -> float:
        """
        Find the exit price for a trade.
        
        Parameters:
        -----------
        trade : Trade
            Trade object
        
        """
        try:
            # Look for exact match of exit date
            exit_data = self.market_data[
                (self.market_data['QUOTE_DATE'] == trade.exit_date) &
                (self.market_data['STRIKE'] == trade.strike)
            ]
            
            # If no exact match, try to find the closest date
            if exit_data.empty:
                print(f"Warning: No exact match for exit date {trade.exit_date.date()} with strike {trade.strike}")
                
                # Filter by strike
                alternative_data = self.market_data[self.market_data['STRIKE'] == trade.strike].copy()
                if alternative_data.empty:
                    print(f"  Skipping trade - no suitable exit data found")
                    return None
                
                # Find closest date before exit date
                alternative_data = alternative_data[alternative_data['QUOTE_DATE'] <= trade.exit_date]
                if alternative_data.empty:
                    print(f"  Skipping trade - no suitable exit data found before exit date")
                    return None
                
                closest_date = alternative_data['QUOTE_DATE'].max()
                
                # Only use if within a reasonable range (e.g., 5 days)
                max_days_diff = 5
                days_diff = (trade.exit_date - closest_date).days
                if days_diff > max_days_diff:
                    print(f"  Skipping trade - closest date too far ({days_diff} days)")
                    return None
                
                print(f"  Using {closest_date.date()} instead (closest available)")
                exit_data = self.market_data[
                    (self.market_data['QUOTE_DATE'] == closest_date) &
                    (self.market_data['STRIKE'] == trade.strike)
                ]
            
            if exit_data.empty:
                return None
            
            # Extract useful columns for determining price
            underlying_price = None
            if 'UNDERLYING_PRICE' in exit_data.columns and not pd.isna(exit_data['UNDERLYING_PRICE'].iloc[0]):
                underlying_price = exit_data['UNDERLYING_PRICE'].iloc[0]
            elif 'UNDERLYING_LAST' in exit_data.columns and not pd.isna(exit_data['UNDERLYING_LAST'].iloc[0]):
                underlying_price = exit_data['UNDERLYING_LAST'].iloc[0]
            
            if underlying_price is None:
                print("  Warning: No underlying price found in market data")
                return None
            
            if trade.position_type in ['put', 'SHORT_PUT']:
                # For put options
                if trade.exit_date >= trade.entry_date.replace(month=trade.entry_date.month+1):
                    # If exit date is at least a month away, use mid price
                    if 'P_BID' in exit_data.columns and 'P_ASK' in exit_data.columns:
                        bid = exit_data['P_BID'].iloc[0]
                        ask = exit_data['P_ASK'].iloc[0]
                        if not pd.isna(bid) and not pd.isna(ask):
                            return (bid + ask) / 2
                
                # Calculate intrinsic value for put
                intrinsic_value = max(trade.strike - underlying_price, 0)
                
                # Add a small time value for options not at expiration
                days_to_expiry = (trade.exit_date - trade.entry_date).days
                if days_to_expiry > 0:
                    time_value_factor = min(days_to_expiry / 365, 0.2)  # Cap at 20%
                    time_value = intrinsic_value * time_value_factor
                    return intrinsic_value + time_value
                
                return intrinsic_value
            else:  # call options
                # For call options, try to use market prices first
                if 'C_BID' in exit_data.columns and 'C_ASK' in exit_data.columns:
                    bid = exit_data['C_BID'].iloc[0]
                    ask = exit_data['C_ASK'].iloc[0]
                    if not pd.isna(bid) and not pd.isna(ask):
                        return (bid + ask) / 2
                
                # Calculate intrinsic value for call
                intrinsic_value = max(underlying_price - trade.strike, 0)
                
                # Add a small time value for options not at expiration
                days_to_expiry = (trade.exit_date - trade.entry_date).days
                if days_to_expiry > 0:
                    time_value_factor = min(days_to_expiry / 365, 0.2)  # Cap at 20%
                    time_value = intrinsic_value * time_value_factor
                    return intrinsic_value + time_value
                
                return intrinsic_value
        except Exception as e:
            print(f"  Error finding exit price: {e}")
            return None
    
    def _calculate_trade_pnl(self, trade: Trade, exit_price: float) -> float:
        """
        Calculate the P&L for a trade.
        
        Parameters:
        -----------
        trade : Trade
            Trade object
        exit_price : float
            Exit price
            
        
        """
        if trade.action in ['buy', 'BUY']:
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # sell
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        return pnl
    
    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from the backtest results.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with trade results
        
        """
        if results_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Only include executed trades (not skipped due to drawdown limits)
        executed_df = results_df[results_df['status'] == 'executed']
        
        # Basic metrics
        total_trades = len(executed_df)
        profitable_trades = executed_df[executed_df['pnl'] > 0]
        losing_trades = executed_df[executed_df['pnl'] <= 0]
        
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        total_pnl = executed_df['pnl'].sum()
        avg_pnl = executed_df['pnl'].mean() if total_trades > 0 else 0
        max_profit = executed_df['pnl'].max() if total_trades > 0 else 0
        max_loss = executed_df['pnl'].min() if total_trades > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = profitable_trades['pnl'].sum() if not profitable_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor
        }
    
    def calculate_cagr(self, equity_curve: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                     initial_capital: float) -> float:
        """
        Calculate the Compound Annual Growth Rate (CAGR).
        
        Parameters:
        -----------
        equity_curve : pandas.Series
            Series with equity values
        start_date : pd.Timestamp
            Start date of the backtest
        end_date : pd.Timestamp
            End date of the backtest
        initial_capital : float
            Initial capital
        
        """
        # If equity curve is empty or no time has passed, return 0
        if equity_curve.empty or start_date == end_date:
            return 0.0
        
        # Get the final equity value
        final_equity = equity_curve.iloc[-1]
        
        # Calculate years
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Calculate CAGR
        if years > 0 and initial_capital > 0 and final_equity > 0:
            cagr = (final_equity / initial_capital) ** (1 / years) - 1
            return cagr
        else:
            return 0.0 