import pandas as pd
import numpy as np

signals = pd.read_csv('data/signals_mispricing.csv')
market_data = pd.read_csv('data/clean/SPY_2023_eod.csv')

keys = ['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE']
merged_df = pd.merge(signals, market_data, on=keys, how='inner')

trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE', 'action', 'position_type', 'confidence']]
trade_data['entry_price'] = merged_df[['P_BID', 'P_ASK']].mean(axis=1)

#build a class for trades
class Trade:
    
    def __init__(self, entry_date, exit_date, strike, action, position_type, entry_price, confidence=None, quantity=1):
        """
        Represents a single option trade.
        
        Parameters:
        -----------
        entry_date : str or datetime
            Date when the trade is entered
        exit_date : str or datetime
            Date when the trade is exited
        strike : float
            Strike price of the option
        action : str
            'buy' or 'sell'
        position_type : str
            'call' or 'put'
        entry_price : float
            Price at which the trade is entered
        confidence : float, optional
            Confidence score (if available)
        quantity : int, optional
            Number of contracts to trade (default 1)
        """
        self.entry_date = pd.to_datetime(entry_date) if isinstance(entry_date, str) else entry_date
        self.exit_date = pd.to_datetime(exit_date) if isinstance(exit_date, str) else exit_date
        self.strike = float(strike)
        self.action = action
        self.position_type = position_type
        self.entry_price = float(entry_price)
        self.confidence = confidence
        self.quantity = int(quantity)
        
    def is_active(self, date: str):
        return self.entry_date <= date <= self.exit_date
    
    def calc_exit_price(self, underlying_price):
        """
        Calculate the exit price for the option.
        
        Parameters:
        -----------
        underlying_price : float
            Price of the underlying asset at exit
            
        Returns:
        --------
        float
            Exit price of the option
        """
        if self.position_type == 'put':
            # For put options, value = max(Strike - Underlying, 0)
            return max(self.strike - underlying_price, 0)
        else:
            # For call options, would need more complex pricing model
            return 0  # Placeholder

#build a class to store trades
class StoreTrades:
    def __init__(self):
        """
        Container for storing and managing a collection of trades.
        """
        self.trades = []
        
    def add_trade(self, trade):
        """
        Add a trade to the collection.
        
        Parameters:
        -----------
        trade : Trade
            Trade object to add
        """
        self.trades.append(trade)
        
    def get_trades(self):
        """
        Get all trades in the collection.
        
        Returns:
        --------
        list
            List of Trade objects
        """
        return self.trades
    
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
        quantity=1  # or dynamic later
    )
    trades.add_trade(trade)

        