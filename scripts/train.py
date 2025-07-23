import yaml
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Adjust path to import from the parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rlfd.data_loader import load_and_clean_data
from rlfd.preprocessing import preprocess_features, create_windows
from rlfd.models import DQNAgentLSTM, DQNAgentTransformer
from rlfd.trainer import train_agent, evaluate_model
from rlfd.utils import set_seed, save_results_to_csv

def main():
    # --- 1. Load Configuration ---
    with open('config/params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    prep_config = config['preprocessing']
    model_config = config['model']
    train_config = config['training']
    
    # --- 2. Set Up Environment ---
    if train_config['seed']:
        set_seed(train_config['seed'])
        print(f"Using fixed seed: {train_config['seed']}")
    else:
        # Generate a random seed if none is provided
        seed = random.randint(0, 100000)
        train_config['seed'] = seed # Store seed for logging
        set_seed(seed)
        print(f"Using random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load and Preprocess Data ---
    df_cleaned = load_and_clean_data(data_config['path'])
    df_ids, features_df = preprocess_features(df_cleaned, prep_config['top_n_categorical'], prep_config['top_m_values'])
    windows, masks, labels, bt_ids = create_windows(df_ids, features_df, prep_config['window_size'])
    
    # --- 4. Split Data ---
    (windows_remain, test_windows, masks_remain, test_masks, 
     labels_remain, test_labels, bt_ids_remain, test_bt_ids) = train_test_split(
        windows, masks, labels, bt_ids, 
        test_size=prep_config['test_split_size'], 
        random_state=train_config['seed'], 
        stratify=labels
    )
    
    (train_windows, val_windows, train_masks, val_masks, 
     train_labels, val_labels, train_bt_ids, val_bt_ids) = train_test_split(
        windows_remain, masks_remain, labels_remain, bt_ids_remain,
        test_size=prep_config['validation_split_size'],
        random_state=train_config['seed'],
        stratify=labels_remain
    )
    
    print(f"Train set: {len(train_windows)} | Validation set: {len(val_windows)} | Test set: {len(test_windows)}")

    # --- 5. Initialize Model ---
    input_size = train_windows.shape[2]
    num_actions = 2
    if model_config['type'] == 'lstm':
        agent = DQNAgentLSTM(input_size, model_config['hidden_size'], num_actions)
    elif model_config['type'] == 'transformer':
        agent = DQNAgentTransformer(
            input_size, model_config['hidden_size'], num_actions,
            model_config['n_heads'], model_config['num_layers'], model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    agent.to(device)

    # --- 6. Train Model ---
    train_data = {'windows': train_windows, 'masks': train_masks, 'labels': train_labels, 'bt_ids': train_bt_ids}
    val_data = {'windows': val_windows, 'masks': val_masks, 'labels': val_labels}
    
    trained_agent, best_model_state = train_agent(agent, device, train_data, val_data, train_config)
    
    # --- 7. Save Best Model and Evaluate ---
    if best_model_state:
        save_path = data_config.get('model_save_path')
        if save_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the model state dictionary
            torch.save(best_model_state, save_path)
            print(f"Best model state saved to {save_path}")
        else:
            print("No 'model_save_path' found in config. Skipping model save.")
            
    test_data = {'windows': test_windows, 'masks': test_masks, 'labels': test_labels}
    final_results, best_results = evaluate_model(trained_agent, device, test_data, best_model_state)

    # --- 8. Log Results to CSV ---
    log_params = {**prep_config, **model_config, **train_config}
    
    # Save results for the final model state
    preds, acc, _, cm = final_results
    save_results_to_csv(acc, test_labels, preds, cm, {**log_params, 'model_version': 'final_epoch'}, data_config['results_log_file'])

    # Save results for the best validation model if it exists
    if best_results:
        preds, acc, _, cm = best_results
        save_results_to_csv(acc, test_labels, preds, cm, {**log_params, 'model_version': 'best_validation'}, data_config['results_log_file'])

if __name__ == '__main__':
    main()