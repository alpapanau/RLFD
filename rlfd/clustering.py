import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

def perform_dbscan_clustering(features_df: pd.DataFrame, df_ids: pd.DataFrame, 
                              selected_features: List[str], eps: float, min_samples: int) -> Tuple[np.ndarray, Dict]:
    """
    Clusters the data using DBSCAN, computes fraud rates per cluster, and visualizes the results.

    Args:
        features_df (pd.DataFrame): The processed feature set.
        df_ids (pd.DataFrame): DataFrame containing labels, aligned with features_df.
        selected_features (List[str]): Names of features from features_df to use for clustering.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        Tuple[np.ndarray, Dict]: A tuple containing:
            - cluster_labels (np.ndarray): Array of cluster assignments for each sample.
            - fraud_rates (Dict): A dictionary mapping each cluster label to its fraud rate.
    """
    available_features = [feat for feat in selected_features if feat in features_df.columns]
    if not available_features:
        raise ValueError("None of the selected features for clustering are available in the features DataFrame.")
    print(f"Clustering using {len(available_features)} features: {available_features}")

    data_for_clustering = features_df[available_features]
    data_scaled = StandardScaler().fit_transform(data_for_clustering)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    cluster_labels = dbscan.fit_predict(data_scaled)

    # --- Analysis and Visualization ---
    df_analysis = df_ids.copy()
    df_analysis['cluster'] = cluster_labels

    fraud_summary = df_analysis.groupby('cluster')['bonifico.last_status'].agg(
        total='count',
        frauds='sum'
    ).reset_index()
    fraud_summary['fraud_rate'] = fraud_summary['frauds'] / fraud_summary['total']

    print("\n--- Fraud Rate per Cluster ---")
    print(fraud_summary[['cluster', 'total', 'frauds', 'fraud_rate']].round(4))



    fraud_rates = dict(zip(fraud_summary['cluster'], fraud_summary['fraud_rate']))
    return cluster_labels, fraud_rates

def defragment_bt_clusters(df_with_clusters: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """
    Reassigns short cluster fragments within a client's transaction history to the dominant neighboring clusters.
    This helps to smooth out noisy cluster assignments over time.

    Args:
        df_with_clusters (pd.DataFrame): DataFrame containing 'cluster' and 'bonifico.prodotto_bt' columns.
        max_gap (int): The maximum length of a fragment to be considered for reassignment.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'cluster_defragmented' column.
    """
    print(f"Defragmenting cluster assignments with max_gap={max_gap}...")
    df = df_with_clusters.copy().sort_index() # Ensure chronological order by index
    df['cluster_defragmented'] = df['cluster']

    for bt, group in df.groupby('bonifico.prodotto_bt'):
        if len(group) <= max_gap:
            continue

        clusters = group['cluster_defragmented'].values
        
        # Iteratively smooth out small fragments
        for _ in range(2): # Run twice for better smoothing
            i = 0
            while i < len(clusters):
                current_cluster = clusters[i]
                j = i
                while j < len(clusters) and clusters[j] == current_cluster:
                    j += 1
                
                # We found a fragment of `current_cluster` from index `i` to `j-1`
                fragment_len = j - i
                
                if 0 < fragment_len <= max_gap:
                    prev_cluster = clusters[i-1] if i > 0 else None
                    next_cluster = clusters[j] if j < len(clusters) else None
                    
                    # If fragment is surrounded by the same cluster, merge it
                    if prev_cluster is not None and prev_cluster == next_cluster:
                        clusters[i:j] = prev_cluster
                i = j
        
        df.loc[group.index, 'cluster_defragmented'] = clusters
        
    return df