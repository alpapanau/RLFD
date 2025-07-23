import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from typing import Tuple, List

def preprocess_features(df: pd.DataFrame, top_n: int, top_m: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the raw dataframe to create scaled numerical features.

    Args:
        df (pd.DataFrame): The input dataframe from the data loader.
        top_n (int): The number of top categorical features to select based on mutual information.
        top_m (int): For each selected categorical feature, keeps the top M most frequent values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_ids: A DataFrame with client ID ('bonifico.prodotto_bt') and label.
            - features_df: A DataFrame with the final, scaled features.
    """
    print("Starting feature preprocessing...")
    # Sort by client and time for sequential processing
    df = df.sort_values(by=['bonifico.prodotto_bt', 'bonifico.prodotto_dataora']).reset_index(drop=True)

    # Calculate distance from each client's spatial median transaction location
    median_coords = df.groupby('bonifico.prodotto_bt')[['bonifico.prodotto_latitude', 'bonifico.prodotto_longitude']].median()
    df = df.join(median_coords, on='bonifico.prodotto_bt', rsuffix='_median')
    df['distance_from_median'] = np.sqrt(
        (df['bonifico.prodotto_latitude'] - df['bonifico.prodotto_latitude_median'])**2 +
        (df['bonifico.prodotto_longitude'] - df['bonifico.prodotto_longitude_median'])**2
    )
    df = df.drop(columns=['bonifico.prodotto_latitude_median', 'bonifico.prodotto_longitude_median'])

    # Separate identifiers and labels from features
    df_ids = df[['bonifico.prodotto_bt', 'bonifico.last_status']].copy()
    features_df = df.drop(columns=['bonifico.prodotto_bt', 'bonifico.prodotto_dataora', 'bonifico.last_status'])

    # --- Categorical Feature Handling ---
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    
    # Select top N categorical features using mutual information
    if len(categorical_cols) > 0 and top_n > 0:
        cat_for_mi = features_df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)
        mi_scores = mutual_info_classif(cat_for_mi, df_ids['bonifico.last_status'], discrete_features=True)
        top_n_features = pd.Series(mi_scores, index=categorical_cols).nlargest(top_n).index.tolist()
        print(f"Selected top {len(top_n_features)} categorical features: {top_n_features}")
        
        # Reduce cardinality of selected features to top M values
        for col in top_n_features:
            top_m_values = features_df[col].value_counts().nlargest(top_m).index
            features_df[col] = features_df[col].where(features_df[col].isin(top_m_values), 'Other')
            
        # Drop non-selected categorical columns
        cols_to_drop = [col for col in categorical_cols if col not in top_n_features]
        features_df = features_df.drop(columns=cols_to_drop)
        
        # One-hot encode the final categorical features
        features_df = pd.get_dummies(features_df, drop_first=True)

    # Scale all features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
    
    print(f"Preprocessing complete. Final feature shape: {features_df.shape}")
    return df_ids, features_df

def create_windows(df_ids: pd.DataFrame, features_df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates padded, sliding windows of transaction sequences for each client.

    Args:
        df_ids (pd.DataFrame): DataFrame with client IDs and labels.
        features_df (pd.DataFrame): DataFrame with scaled features.
        window_size (int): The length of each sequence window.

    Returns:
        Tuple containing windows, masks, labels, and client IDs for each window.
    """
    print(f"Creating windows of size {window_size}...")
    windows, masks, labels, bt_ids = [], [], [], []

    df_ids = df_ids.copy()
    df_ids['original_index'] = df_ids.index

    for bt, group in df_ids.groupby('bonifico.prodotto_bt'):
        indices = group['original_index'].values
        client_features = features_df.loc[indices].values
        client_labels = group['bonifico.last_status'].values
        n_samples = len(client_features)

        if n_samples == 0:
            continue

        # Padding for clients with fewer transactions than window_size
        if n_samples < window_size:
            pad_len = window_size - n_samples
            # Use a distinct padding value like -10 that is outside the scaled range [0, 1]
            padding = np.full((pad_len, client_features.shape[1]), -10.0)
            window = np.vstack([client_features, padding])
            mask = [1] * n_samples + [0] * pad_len
            label = client_labels[-1] # Label of the last available transaction
            windows.append(window)
            masks.append(mask)
            labels.append(label)
            bt_ids.append(bt)
        else:
            # Create sliding windows for clients with enough transactions
            for i in range(n_samples - window_size + 1):
                window = client_features[i : i + window_size]
                mask = [1] * window_size
                label = client_labels[i + window_size - 1] # Label of the last transaction in the window
                windows.append(window)
                masks.append(mask)
                labels.append(label)
                bt_ids.append(bt)

    print(f"Created {len(windows)} windows.")
    return np.array(windows), np.array(masks), np.array(labels), np.array(bt_ids)