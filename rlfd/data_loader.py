import os
import pandas as pd
import numpy as np
from typing import Tuple

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Loads all CSV files from a directory, concatenates them, and performs initial cleaning.

    Args:
        path (str): The path to the directory containing CSV files.

    Returns:
        pd.DataFrame: A single, cleaned DataFrame.
    """
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {path}")

    df_list = []
    total_rows = 0
    total_fraud = 0

    print("Loading data...")
    for file in all_files:
        df = pd.read_csv(file, sep="§", engine="python")
        fraud_count = np.sum(df['bonifico.last_status'] == 1)
        total_rows += len(df)
        total_fraud += fraud_count
        
        print(f"  - Loaded {os.path.basename(file)}: {len(df)} rows, "
              f"{100 * fraud_count / len(df):.2f}% fraud")
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True, sort=False)
    print(f"\nTotal rows loaded: {total_rows}")
    print(f"Overall fraud percentage: {100 * total_fraud / total_rows:.2f}%\n")

    print("Dropping duplicates and performing initial feature engineering...")
    # Drop duplicates, keeping the last transaction record
    df_cleaned = full_df.drop_duplicates(subset='bonifico.prodotto_x_isp_chiaveantifrode', keep='last').copy()
    
    # --- Initial Feature Engineering ---
    # 1. Country Code
    df_cleaned['CountryCodeBIC_isIT'] = df_cleaned['CountryCodeBIC'] == 'f415bf7b07a9b2c07029144aafb3c59d0187682ecd2b8c8ac911e742a38a5f36'

    # 2. Weekend/Weekday
    df_cleaned['bonifico.prodotto_dataora'] = pd.to_datetime(df_cleaned['bonifico.prodotto_dataora'])
    df_cleaned['isWeekEnd'] = df_cleaned['bonifico.prodotto_dataora'].dt.weekday.isin([5, 6])

    # 3. Hour of the day
    df_cleaned['hour'] = df_cleaned['bonifico.prodotto_dataora'].dt.hour
    
    print(f"Dataframe shape after cleaning: {df_cleaned.shape}")
    return df_cleaned