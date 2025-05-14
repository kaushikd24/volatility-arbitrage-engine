import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

options_df = pd.read_csv('data/clean/SPY_2023_eod.csv', index_col=0, parse_dates=True)

def moneyness(strike, underlying_last):
    moneyness = strike/underlying_last
    return moneyness

options_df['MONEYNESS'] = options_df.apply(lambda row: moneyness(row['STRIKE'], row['UNDERLYING_LAST']), axis=1)
date_strings = options_df.index.strftime('%Y-%m-%d')
dates_unique = date_strings.unique()
dates = dates_unique.tolist()

options_df = options_df[['EXPIRE_DATE', 'STRIKE', 'C_IV', 'P_IV', 'UNDERLYING_LAST', 'MONEYNESS']]

expire_dates = options_df['EXPIRE_DATE'].unique()
expire_dates = expire_dates.tolist()

class Smile:
    
    def __init__(self, quote_date, expire_date, df):
        self.df = df
        self.quote_date = quote_date
        self.expire_date = expire_date
        
    def plot_smile(self):
        plt.plot(self.df['MONEYNESS'], self.df['C_IV'], label = 'CALL IV')
        plt.plot(self.df['MONEYNESS'], self.df['P_IV'], label = 'PUT IV')
        plt.xlabel('MONEYNESS')
        plt.ylabel('IV')
        plt.title(f'VOLATILITY SMILE: {self.quote_date} - {self.expire_date}')
        plt.legend()
        plt.show()

class VolatilitySurface:
    def __init__(self, quote_date):
        self.quote_date = quote_date
        self.smiles = {}
        
    def add_smile(self, smile_obj):
        self.smiles[smile_obj.expire_date] = smile_obj
    
    def get_smile(self, expire_date):
        return self.smiles.get(expire_date)
    

class VolSurfaceStore:
    def __init__(self):
        self.surfaces = {}
        
    def add_surface(self, smile_list):
        for smile in smile_list:
            q = smile.quote_date
            if q not in self.surfaces:
                self.surfaces[q] = VolatilitySurface(q)
            self.surfaces[q].add_smile(smile)
            
    def get_surface(self, quote_date):
        return self.surfaces[quote_date]

smiles = []
for q in dates:
    for e in expire_dates:
        temp = options_df.loc[q]
        temp = temp[temp['EXPIRE_DATE'] == e]
        smiles.append(Smile(q, e, temp))
        del temp

VS = VolSurfaceStore()
VS.add_surface(smiles)

    


    