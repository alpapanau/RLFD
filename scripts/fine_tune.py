import yaml
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Adjust path to import from the project root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rlfd.data_loader import load_and_clean_data
from rlfd.preprocessing import preprocess_features, create_windows
from rlfd.models import DQNAgentLSTM, DQNAgentTransformer
from rlfd.trainer import evaluate_model # We only need evaluate_model from the trainer
from rlfd.utils import set_seed, save_results_to_csv

def fine_tune_on_uncertain(agent, device, windows, masks, labels, config):
    """
    Identifies uncertain samples from the training set and fine-tunes the agent on them
    using a supervised classification approach.
    (This function is identical to the one in the previous combined script)
    """
    print("\n--- Identifying Uncertain Samples for Fine-Tuning ---")
    agent.eval()

    ft_config = config['fine_tuning']
    batch_size = config['training']['batch_size']

    uncertainties = []
    with torch.no_grad():
        dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32),
                              torch.tensor(masks.sum(axis=1), dtype=torch.int64))
        loader = DataLoader(dataset, batch_size=batch_size * 4)

        for batch_windows, batch_lengths in loader:
            batch_windows, batch_lengths = batch_windows.to(device), batch_lengths.to(device)
            q_values = agent(batch_windows, batch_lengths)
            q_diffs = torch.abs(q_values[:, 1] - q_values[:, 0])
            uncertainties.extend(q_diffs.cpu().numpy())

    uncertainties = np.array(uncertainties)
    k_percent = ft_config['k_percent_uncertain']
    if not (0 < k_percent < 100):
        print(f"Invalid k_percent_uncertain: {k_percent}. Aborting.")
        return None

    uncertainty_threshold = np.percentile(uncertainties, k_percent)
    uncertain_indices = np.where(uncertainties <= uncertainty_threshold)[0]

    if len(uncertain_indices) == 0:
        print("No uncertain samples found below the threshold. Aborting.")
        return None

    print(f"Identified {len(uncertain_indices)} samples for fine-tuning (top {k_percent}% most uncertain).")

    print("\n--- Starting Targeted Fine-Tuning ---")
    agent.train()

    ft_windows = torch.tensor(windows[uncertain_indices], dtype=torch.float32)
    ft_masks = torch.tensor(masks[uncertain_indices].sum(axis=1), dtype=torch.int64)
    ft_labels = torch.tensor(labels[uncertain_indices], dtype=torch.int64)
    
    ft_dataset = TensorDataset(ft_windows, ft_masks, ft_labels)
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=ft_config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(ft_config['epochs']):
        total_loss = 0
        for batch_windows, batch_lengths, batch_labels in ft_loader:
            batch_windows, batch_lengths, batch_labels = batch_windows.to(device), batch_lengths.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = agent(batch_windows, batch_lengths)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-Tuning Epoch {epoch+1}/{ft_config['epochs']}, Loss: {total_loss/len(ft_loader):.4f}")

    return agent


def main():
    # --- 1. Load Configuration ---
    with open('config/params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    prep_config = config['preprocessing']
    model_config = config['model']
    train_config = config['training']
    ft_config = config['fine_tuning']
    
    # Check if fine-tuning is enabled in the config
    if not config.get('fine_tuning', {}).get('enabled', False):
        print("Fine-tuning is disabled in 'config/params.yaml'. Exiting.")
        return

    # --- 2. Set Up Environment ---
    set_seed(train_config['seed']) # Use the same seed as training for data splits
    print(f"Using fixed seed from config: {train_config['seed']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load and Preprocess Data (Must be identical to training) ---
    print("\n--- Preparing Data ---")
    df_cleaned = load_and_clean_data(data_config['path'])
    df_ids, features_df = preprocess_features(df_cleaned, prep_config['top_n_categorical'], prep_config['top_m_values'])
    windows, masks, labels, bt_ids = create_windows(df_ids, features_df, prep_config['window_size'])
    
    # Recreate the exact same data splits as in train.py
    (windows_remain, test_windows, masks_remain, test_masks, 
     labels_remain, test_labels, bt_ids_remain, _) = train_test_split(
        windows, masks, labels, bt_ids, 
        test_size=prep_config['test_split_size'], 
        random_state=train_config['seed'], 
        stratify=labels
    )
    
    (train_windows, _, train_masks, _, 
     train_labels, _, _, _) = train_test_split(
        windows_remain, masks_remain, labels_remain, bt_ids_remain, 
        test_size=prep_config['validation_split_size'],
        random_state=train_config['seed'],
        stratify=labels_remain
    )
    print(f"Data prepared. Using {len(train_windows)} training samples for uncertainty analysis.")

    # --- 4. Initialize and Load Pre-Trained Model ---
    print("\n--- Loading Pre-Trained Model ---")
    input_size = train_windows.shape[2]
    num_actions = 2
    if model_config['type'] == 'lstm':
        agent = DQNAgentLSTM(input_size, model_config['hidden_size'], num_actions)
    else: # Add other model types if needed
        agent = DQNAgentTransformer(input_size, model_config['hidden_size'], num_actions,
            model_config['n_heads'], model_config['num_layers'], model_config['dropout'])
    
    model_path = data_config.get('model_save_path')
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Pre-trained model not found at '{model_path}'.")
        print("Please run 'train.py' first to generate the model file.")
        return

    # Load the state dictionary from the saved file
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.to(device)
    print(f"Successfully loaded model from {model_path}")

    # --- 5. Run Fine-Tuning ---
    fine_tuned_agent = fine_tune_on_uncertain(
        agent, device, train_windows, train_masks, train_labels, config
    )

    if not fine_tuned_agent:
        print("Fine-tuning did not complete successfully.")
        return

    # --- 6. Evaluate and Save Results ---
    print("\n--- Evaluating Model After Fine-Tuning ---")
    test_data = {'windows': test_windows, 'masks': test_masks, 'labels': test_labels}
    ft_results, _ = evaluate_model(fine_tuned_agent, device, test_data)

    # Save the fine-tuned model state
    ft_save_path = data_config.get('fine_tuned_model_save_path')
    if ft_save_path:
        os.makedirs(os.path.dirname(ft_save_path), exist_ok=True)
        torch.save(fine_tuned_agent.state_dict(), ft_save_path)
        print(f"Fine-tuned model state saved to {ft_save_path}")

    # Log results to CSV
    log_params = {**prep_config, **model_config, **train_config, **config.get('fine_tuning', {})}
    preds_ft, acc_ft, _, cm_ft = ft_results
    save_results_to_csv(acc_ft, test_labels, preds_ft, cm_ft, {**log_params, 'model_version': 'fine_tuned'}, data_config['results_log_file'])

if __name__ == '__main__':
    main()