import pandas as pd
import numpy as np

spy_df = pd.read_csv('data/clean/SPY_2023_eod.csv', index_col = 0, parse_dates = True)
spy_df = pd.read_csv('data/clean/SPY_2023_eod.csv')
spy_df.columns = spy_df.columns.str.strip()         
spy_df.reset_index(inplace=True)                    
spy_df['QUOTE_DATE'] = pd.to_datetime(spy_df['QUOTE_DATE'])
spy_df['EXPIRE_DATE'] = pd.to_datetime(spy_df['EXPIRE_DATE'])

spy_df.drop(columns=['index'], inplace=True)

smile_features = pd.read_csv('data/smile_features.csv')

smile_df = smile_features
smile_df.drop(columns=['Unnamed: 0'], inplace=True)

term_df = pd.read_csv('data/term_structure.csv')

spy_df['moneyness'] = spy_df['STRIKE'] / spy_df['UNDERLYING_LAST']
spy_df['mid_iv'] = 0.5 * (spy_df['C_IV'] + spy_df['P_IV'])
spy_df['mid_delta'] = 0.5 * (spy_df['C_DELTA'] + spy_df['P_DELTA'])
spy_df['mid_gamma'] = 0.5 * (spy_df['C_GAMMA'] + spy_df['P_GAMMA'])
spy_df['mid_theta'] = 0.5 * (spy_df['C_THETA'] + spy_df['P_THETA'])
spy_df['mid_vega'] = 0.5 * (spy_df['C_VEGA'] + spy_df['P_VEGA'])

smile_df.columns = smile_df.columns.str.lower()  # ensure consistent merge keys
merged_df = spy_df.merge(
    smile_df,
    how='left',
    left_on=['QUOTE_DATE', 'EXPIRE_DATE'],
    right_on=['quote_date', 'expire_date']
)

merged_df['QUOTE_DATE'] = pd.to_datetime(merged_df['QUOTE_DATE'])
term_df['quote_date'] = pd.to_datetime(term_df['quote_date'])

merged_df = merged_df.merge(
    term_df,
    how='left',
    left_on='QUOTE_DATE',
    right_on='quote_date'
)

#iv threshold as 0.04
merged_df['label'] = (merged_df['mid_iv'] > merged_df['atm_iv'] + 0.04).astype(int)

merged_df.to_csv('data/features.csv')


