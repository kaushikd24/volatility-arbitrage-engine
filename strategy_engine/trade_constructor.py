import pandas as pd
import numpy as np

signals = pd.read_csv('data/signals_mispricing.csv')
market_data = pd.read_csv('data/clean/SPY_2023_eod.csv')

keys = ['QUOTE_DATE', 'STRIKE', 'EXPIRE_DATE']
merged_df = pd.merge(signals, market_data, on=keys, how='inner')

trade_data = merged_df[['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE','trade_type', 'confidence']]
trade_data['entry_price'] = merged_df[['P_BID'] + ['P_ASK']].mean(axis=1)

#build a class for trades
class Trade:
    
    def __init__(self, entry_date, exit_date, strike, trade_type, entry_price, confidence, quantity=1):
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.strike = strike
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.confidence = confidence
        self.quantity = quantity
        
    def is_active(self, date: str):
        return self.entry_date <= date <= self.exit_date
    
#build a class to store trades
class StoreTrades:
    def __init__(self):
        self.trades = []
        
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        
    def get_trades(self):
        return self.trades
    
trades = StoreTrades()
trades.add_trade(Trade(trade_data['QUOTE_DATE'], trade_data['EXPIRE_DATE'], trade_data['STRIKE'], trade_data['trade_type'], trade_data['entry_price'], trade_data['confidence']))

trades = StoreTrades()

for _, row in trade_data.iterrows():
    trade = Trade(
        entry_date=row['QUOTE_DATE'],
        exit_date=row['EXPIRE_DATE'],
        strike=row['STRIKE'],
        trade_type=row['trade_type'],
        entry_price=row['entry_price'],
        confidence=row['confidence'],
        quantity=1  # or dynamic later
    )
    trades.add_trade(trade)

        