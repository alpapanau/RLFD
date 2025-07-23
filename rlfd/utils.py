import os
import csv
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_results_to_csv(accuracy: float, true_labels: np.ndarray, predicted_labels: np.ndarray, 
                        cm: np.ndarray, params: dict, filename: str):
    """
    Saves the evaluation metrics and hyperparameters to a CSV file.

    Args:
        accuracy (float): Overall accuracy.
        true_labels (np.ndarray): Ground truth labels.
        predicted_labels (np.ndarray): Model's predictions.
        cm (np.ndarray): Confusion matrix.
        params (dict): Dictionary of hyperparameters used for the run.
        filename (str): Path to the output CSV file.
    """
    report_dict = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)

    flat_report = {
        "precision_0": report_dict.get("0", {}).get("precision"),
        "recall_0": report_dict.get("0", {}).get("recall"),
        "f1_0": report_dict.get("0", {}).get("f1-score"),
        "precision_1": report_dict.get("1", {}).get("precision"),
        "recall_1": report_dict.get("1", {}).get("recall"),
        "f1_1": report_dict.get("1", {}).get("f1-score"),
        "accuracy": accuracy,
        "confusion_matrix": str(cm).replace('\n', ' ')
    }
    
    row_data = {**flat_report, **params}
    df_row = pd.DataFrame([row_data])

    file_exists = os.path.exists(filename)
    df_row.to_csv(filename, mode='a', header=not file_exists, index=False)
    print(f"Results saved to {filename}")