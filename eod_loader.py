# eod_loader.py

import pandas as pd
import os


RAW_PATH = "data/options.csv"  
SAVE_DIR = "data/clean"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_and_filter_2023():
    print("Loading CSV...")
    df = pd.read_csv(RAW_PATH, low_memory=False)


    df = df.drop(columns=["Unnamed: 0", "level_0"], errors='ignore')

    df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]

    print(f"Total rows: {len(df):,}")

    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
    df_2023 = df[df["QUOTE_DATE"].dt.year == 2023].copy()

    print(f"Filtered to 2023: {len(df_2023):,} rows")


    keep_cols = [
        "QUOTE_DATE", "QUOTE_TIME_HOURS", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE",
        "C_BID", "C_ASK", "C_IV", "C_DELTA", "C_GAMMA", "C_THETA", "C_VEGA", "C_RHO", "C_VOLUME",
        "P_BID", "P_ASK", "P_IV", "P_DELTA", "P_GAMMA", "P_THETA", "P_VEGA", "P_RHO", "P_VOLUME",
        "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"
    ]
    df_2023 = df_2023[keep_cols]

    save_path = os.path.join(SAVE_DIR, "SPY_2023_eod.csv")
    df_2023.to_csv(save_path, index=False)
    print(f"Saved cleaned file to {save_path}")


if __name__ == "__main__":
    load_and_filter_2023()
