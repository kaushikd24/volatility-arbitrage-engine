import pandas as pd
import numpy as np

from vol_surface.vol_surface_builder import dates, expire_dates

smile_features = pd.read_csv('data/smile_features.csv', index_col = 0)
smile_features.set_index(['quote_date', 'expire_date'], inplace = True)

term_structure_features = []
smile_features['quote_date'] = pd.to_datetime(smile_features['quote_date'])
smile_features['expire_date'] = pd.to_datetime(smile_features['expire_date'])

smile_features.reset_index(inplace = True)
grouped = smile_features.groupby('quote_date')

for quote_date, group in grouped:
    group['dte'] = (group['expire_date'] - quote_date).dt.days
    group = group[group['dte']>0]
    
    if len(group) > 5:
        x = group['dte'].values
        x = x/30 #so that curvature is not very low
        y =group['atm_iv'].values
        
        coeffs = np.polyfit(x,y,2)
        poly= np.poly1d(coeffs)
        
        iv_slope = poly.deriv()(30/30)
        iv_curvature = poly.deriv(2)(30/30)
        short_term_iv = poly(7/30)
        long_term_iv = poly(90/30)
        term_spread = long_term_iv - short_term_iv
        
        term_structure_features.append({
            'quote_date': quote_date,
            'iv_slope': iv_slope,
            'iv_curvature': iv_curvature,
            'short_term_iv': short_term_iv,
            'long_term_iv': long_term_iv,
            'term_spread': term_spread
        })
        
term_structure_features = pd.DataFrame(term_structure_features)
term_structure_features.set_index('quote_date', inplace=True)
term_structure = term_structure_features
term_structure.to_csv('data/term_structure.csv')