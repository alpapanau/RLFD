import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import defaultdict, deque
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any

def _evaluate_on_validation(agent: nn.Module, device: torch.device, val_windows: np.ndarray, 
                           val_masks: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Internal function to evaluate the model on the validation set."""
    agent.eval()
    with torch.no_grad():
        windows_tensor = torch.tensor(val_windows, dtype=torch.float32).to(device)
        lengths = torch.tensor(val_masks.sum(axis=1), dtype=torch.int64).to(device)
        
        q_values = agent(windows_tensor, lengths)
        predicted_labels = torch.argmax(q_values, dim=1).cpu().numpy()

    cm = confusion_matrix(val_labels, predicted_labels, labels=[0, 1])
    report = classification_report(val_labels, predicted_labels, output_dict=True, zero_division=0)
    
    recall_0 = report.get("0", {}).get("recall", 0.0)
    recall_1 = report.get("1", {}).get("recall", 0.0)
    
    return recall_0, recall_1, cm

def train_agent(agent: nn.Module, device: torch.device, train_data: dict, val_data: dict, train_config: Dict[str, Any]):
    """
    Trains the DQN agent using the RLFD methodology.
    
    Args:
        agent (nn.Module): The DQN model to train.
        device (torch.device): The device to train on (CPU or CUDA).
        train_data (dict): Dictionary containing training windows, masks, labels, and client IDs.
        val_data (dict): Dictionary containing validation data.
        train_config (Dict[str, Any]): The 'training' section of the config dictionary.
    """
    if agent.model_type == "lstm":
        optimizer = optim.Adam(agent.parameters(), lr=train_config['learning_rate_lstm'])
    elif agent.model_type == "transformer":
        optimizer = optim.Adam(agent.parameters(), lr=train_config['learning_rate_transformer'])
    else:
        raise ValueError(f"Unknown agent type: {agent.model_type}")

    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=train_config['replay_buffer_size'])
    
    train_windows, train_masks, train_labels, train_bt_ids = train_data.values()
    val_windows, val_masks, val_labels = val_data['windows'], val_data['masks'], val_data['labels']

    bt_to_indices = defaultdict(list)
    for i, bt in enumerate(train_bt_ids):
        bt_to_indices[bt].append(i)

    episodic_bts = [bt for bt, idxs in bt_to_indices.items() if len(idxs) > 1]
    random.shuffle(episodic_bts)
    
    num_episodes = min(len(episodic_bts), train_config['num_episodes'])
    print(f"Starting training for {num_episodes} episodes...")
    
    target_agent = copy.deepcopy(agent).to(device)
    best_recall_1 = 0.0
 

    r1 = train_config['r1_positive_fraud']
    r2 = train_config['r2_positive_normal']
    
    agent.train() 

    for episode in range(num_episodes):
        if not episodic_bts: break
        bt = episodic_bts[episode % len(episodic_bts)]
        indices = bt_to_indices[bt]
        
        epsilon = max(train_config['epsilon_min'], 1.0 - (episode / num_episodes))

        for ie in range(train_config['inner_epochs']):
            i = random.choice(indices)
            state = torch.tensor(train_windows[i], dtype=torch.float32).unsqueeze(0).to(device)
            length = torch.tensor([sum(train_masks[i])], dtype=torch.int64).to(device)

            i_idx = indices.index(i)
            next_i = indices[i_idx + 1] if i_idx < len(indices) - 1 else i
            next_state = torch.tensor(train_windows[next_i], dtype=torch.float32).unsqueeze(0).to(device)
            #epsilon = max(train_config['epsilon_min'], 1.0 - (ie / train_config['inner_epochs']))
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_values = agent(state, length)
                    action = torch.argmax(q_values, dim=1).item()
            
            label = train_labels[i]
            reward = (r1 if action == 1 else -r1) if label == 1 else (r2 if action == 0 else -r2)
            experience = (state, action, reward, next_state, length)
            replay_buffer.append(experience)

            if len(replay_buffer) >= train_config['batch_size']:
                batch = random.sample(replay_buffer, train_config['batch_size'])
                states, actions, rewards, next_states, lengths = zip(*batch)

                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                seq_lengths = torch.tensor([l.item() for l in lengths], dtype=torch.int64).to(device)

                q_current = agent(states, seq_lengths).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    q_next = target_agent(next_states, seq_lengths).max(dim=1)[0]
                target = rewards + train_config['gamma'] * q_next

                loss = criterion(q_current, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode > 0 and episode % train_config['target_update_freq'] == 0:
            target_agent.load_state_dict(agent.state_dict())

        if episode > 0 and episode % train_config['validation_freq'] == 0:
            recall_0, recall_1, _ = _evaluate_on_validation(agent, device, val_windows, val_masks, val_labels)
            print(f"Ep {episode}/{num_episodes} | Val Recall(N/A): {recall_0:.3f}/{recall_1:.3f} | Epsilon: {epsilon:.3f}")

            if recall_1 >= best_recall_1 and recall_0 >= 0.90:
                best_recall_1 = recall_1
                best_model_state = copy.deepcopy(agent.state_dict())
                VEP = episode
                print(f"  -> New best model saved! Best Recall_1: {best_recall_1:.3f}")

            agent.train()          

    print("Training finished.")
    return agent, best_model_state


def evaluate_model(agent: nn.Module, device: torch.device, test_data: dict, best_model_state: dict = None):
    """Evaluates the final and best models on the test set."""
    
    def _evaluate(model_to_eval):
        model_to_eval.eval()
        with torch.no_grad():
            windows = torch.tensor(test_data['windows'], dtype=torch.float32).to(device)
            masks = torch.tensor(test_data['masks'].sum(axis=1), dtype=torch.int64).to(device)
            
            q_values = model_to_eval(windows, masks)
            preds = torch.argmax(q_values, dim=1).cpu().numpy()

        accuracy = accuracy_score(test_data['labels'], preds)
        report_str = classification_report(test_data['labels'], preds, zero_division=0)
        cm = confusion_matrix(test_data['labels'], preds, labels=[0, 1])
        return preds, accuracy, report_str, cm

    print("\n--- Evaluating Final Model on Test Set ---")
    
    final_results = _evaluate(agent)
    
    preds, acc, report_str, cm = final_results # Unpack for printing
    print(f"Accuracy: {acc:.4f}\nClassification Report:\n{report_str}\nConfusion Matrix:\n{cm}")

    best_results = None
    if best_model_state:
        print("\n--- Evaluating Best Validation Model on Test Set ---")
        best_agent = copy.deepcopy(agent)
        best_agent.load_state_dict(best_model_state)
        
        best_results = _evaluate(best_agent)

        preds_best, acc_best, report_best, cm_best = best_results # Unpack for printing
        print(f"Accuracy: {acc_best:.4f}\nClassification Report:\n{report_best}\nConfusion Matrix:\n{cm_best}")

    return final_results, best_results