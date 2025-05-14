import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from vol_surface.vol_surface_builder import VolSurfaceStore
from vol_surface.vol_surface_builder import dates, expire_dates, options_df, VS

smile_features = []

for quote_dates in dates:
    surface = VS.get_surface(quote_dates)
    for expire_date in expire_dates:
        smile_df = surface.get_smile(expire_date).df
        moneyness = np.array(smile_df['MONEYNESS'])
        iv = np.array(0.5*(smile_df['C_IV'] + smile_df['P_IV']))
        mask = ~np.isnan(moneyness) & ~np.isnan(iv)
        x = moneyness[mask]
        y = iv[mask]
        if len(x) > 5:
            coeffs = np.polyfit(x,y,2)
            poly = np.poly1d(coeffs)
            atm_iv = poly(1.0)
            skew = poly.deriv()(1.0)
            curvature = poly.deriv(2)(1.0)
            smile_features.append({
                'quote_date': quote_dates,
                'expire_date': expire_date,
                'atm_iv': atm_iv,
                'skew': skew,
                'curvature': curvature
            })
            
smile_features = pd.DataFrame(smile_features)
smile_features.set_index(['quote_date', 'expire_date'])
smile_features = smile_features.to_csv('data/smile_features.csv')
            
            



