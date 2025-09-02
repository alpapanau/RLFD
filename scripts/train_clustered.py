import yaml
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rlfd.data_loader import load_and_clean_data
from rlfd.preprocessing import preprocess_features, create_windows
from rlfd.clustering import perform_dbscan_clustering, defragment_bt_clusters
from rlfd.models import DQNAgentLSTM, DQNAgentTransformer
from rlfd.trainer import train_agent, evaluate_model
from rlfd.utils import set_seed, save_results_to_csv

def main():
    # --- 1. Load Configuration ---
    with open('config/params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Set Up Environment ---
    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Seed: {config['training']['seed']}")

    # --- 3. Load and Preprocess Data ---
    df_cleaned = load_and_clean_data(config['data']['path'])
    df_ids, features_df = preprocess_features(
    df_cleaned, 
    top_n=config['preprocessing']['top_n_categorical'], 
    top_m=config['preprocessing']['top_m_values']
)

    # --- 4. Perform Clustering ---
    cluster_config = config['clustering_training']
    cluster_labels, fraud_rates = perform_dbscan_clustering(
    features_df, 
    df_ids, 
    selected_features=cluster_config['selected_features'],
    eps=cluster_config['eps'],
    min_samples=cluster_config['min_samples']
    )
    df_ids['cluster'] = cluster_labels
    df_ids = defragment_bt_clusters(df_ids, max_gap=cluster_config['defragment_max_gap'])

    # --- 5. Train and Evaluate an Agent for Each Cluster ---
    # Initialize lists to aggregate predictions from all clusters
    all_true_labels = []
    all_predicted_labels_final = []
    all_predicted_labels_best = []
    
    global_fraud_rate = df_ids['bonifico.last_status'].mean()
    
    cluster_sizes = df_ids['cluster_defragmented'].value_counts()
    total_samples = len(df_ids)
    num_episodes_total = config['training']['num_episodes']

    for cluster_id in cluster_sizes.index:
        
        print(f"\n\n{'='*20} Processing Cluster {cluster_id} {'='*20}")
        
        cluster_share = cluster_sizes[cluster_id] / total_samples
        cluster_num_episodes = max(10, int(num_episodes_total * cluster_share))
        print(f"Cluster Size: {cluster_sizes[cluster_id]}, Episode Budget: {cluster_num_episodes}")

        r1_base = config['training']['r1_positive_fraud']
        cluster_fraud_rate = fraud_rates.get(cluster_id, global_fraud_rate)
        r1_scaled = r1_base * (cluster_fraud_rate / global_fraud_rate) if global_fraud_rate > 0 else r1_base
        print(f"Global Fraud Rate: {global_fraud_rate:.4f}, Cluster Fraud Rate: {cluster_fraud_rate:.4f}, Scaled R1: {r1_scaled:.2f}")
        
        cluster_indices = df_ids[df_ids['cluster_defragmented'] == cluster_id].index
        cluster_ids_df = df_ids.loc[cluster_indices]
        cluster_features_df = features_df.loc[cluster_indices]

        windows, masks, labels, bt_ids = create_windows(cluster_ids_df, cluster_features_df, config['preprocessing']['window_size'])

        (w_rem, w_test, m_rem, m_test, l_rem, l_test, b_rem, _) = train_test_split(
            windows, masks, labels, bt_ids, test_size=0.2, random_state=config['training']['seed'], stratify=labels)
        (w_train, w_val, m_train, m_val, l_train, l_val, b_train, _) = train_test_split(
            w_rem, m_rem, l_rem, b_rem, test_size=0.2, random_state=config['training']['seed'], stratify=l_rem)
            
        agent = DQNAgentLSTM(w_train.shape[2], config['model']['hidden_size'], 2).to(device)
        
        cluster_train_config = config['training'].copy()
        cluster_train_config['num_episodes'] = cluster_num_episodes
        cluster_train_config['r1_positive_fraud'] = r1_scaled

        train_data = {'windows': w_train, 'masks': m_train, 'labels': l_train, 'bt_ids': b_train}
        val_data = {'windows': w_val, 'masks': m_val, 'labels': l_val}
        
        # We need the full config for the trainer to access 'training' section
        full_cluster_config = config.copy()
        full_cluster_config['training'] = cluster_train_config
        
        trained_agent, best_model_state = train_agent(agent, device, train_data, val_data, full_cluster_config['training'])
        
        # --- MODIFIED EVALUATION BLOCK ---
        test_data = {'windows': w_test, 'masks': m_test, 'labels': l_test}
        final_results, best_results = evaluate_model(trained_agent, device, test_data, best_model_state)

        # Unpack predictions from the final model state
        preds_final, _, _, _ = final_results
        
        # Use the "trick": if best_results is None, fall back to final_results
        results_for_best = best_results if best_results is not None else final_results
        preds_best, _, _, _ = results_for_best

        # Aggregate predictions and labels from this cluster's test set
        all_true_labels.extend(l_test)
        all_predicted_labels_final.extend(preds_final)
        all_predicted_labels_best.extend(preds_best)
        # --- END OF MODIFICATION ---

    # --- 6. Final Global Evaluation ---
    log_params = {**config['preprocessing'], **config['model'], **config['training'], **config['clustering_training']}
    
    if all_true_labels:
        # --- Evaluation for FINAL Models ---
        print(f"\n\n{'='*20} GLOBAL EVALUATION (Final Epoch Models) {'='*20}")
        global_accuracy_final = accuracy_score(all_true_labels, all_predicted_labels_final)
        global_report_final = classification_report(all_true_labels, all_predicted_labels_final, zero_division=0)
        global_cm_final = confusion_matrix(all_true_labels, all_predicted_labels_final)

        print(f"Global Accuracy: {global_accuracy_final:.4f}")
        print("Global Classification Report:\n", global_report_final)
        print("Global Confusion Matrix:\n", global_cm_final)
        save_results_to_csv(global_accuracy_final, all_true_labels, all_predicted_labels_final, global_cm_final, 
                            {**log_params, 'model_version': 'clustered_training_final'}, config['data']['results_log_file'])

        # --- Evaluation for BEST Validation Models ---
        print(f"\n\n{'='*20} GLOBAL EVALUATION (Best Validation Models) {'='*20}")
        global_accuracy_best = accuracy_score(all_true_labels, all_predicted_labels_best)
        global_report_best = classification_report(all_true_labels, all_predicted_labels_best, zero_division=0)
        global_cm_best = confusion_matrix(all_true_labels, all_predicted_labels_best)

        print(f"Global Accuracy: {global_accuracy_best:.4f}")
        print("Global Classification Report:\n", global_report_best)
        print("Global Confusion Matrix:\n", global_cm_best)
        save_results_to_csv(global_accuracy_best, all_true_labels, all_predicted_labels_best, global_cm_best, 
                            {**log_params, 'model_version': 'clustered_training_best'}, config['data']['results_log_file'])
    else:
        print("No results to evaluate globally.")

if __name__ == '__main__':
    main()