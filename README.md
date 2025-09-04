# Fraud Detection using Reinforcement Learning (RLFD)

This project implements a Reinforcement Learning-based approach for detecting anomalies in sequential data, such as financial transactions. It uses a Deep Q-Network (DQN) with an LSTM or Transformer backend to learn a policy for classifying transactions as normal or fraudulent.

The project features two alternative training workflows:
1.  **Main Training:** A single, global Reinforcement Learning agent is trained on the entire dataset. The best-performing model based on validation metrics is saved.
2.  **Clustered Training:** This workflow first groups transactions into behavioral clusters and then trains a specialized agent for each cluster.

## Project Structure

- `config/`: Contains the central `params.yaml` configuration file for managing all hyperparameters.
- `data/`: Placeholder for the input datasets. See `data/README.md` for details on where to place your data.
- `models/`: Default directory for saving and loading trained model states (`.pth` files).
- `rlfd/`: The main Python source code package.
  - `clustering.py`: Contains functions for DBSCAN clustering and temporal defragmentation of transaction data.
  - `data_loader.py`: Handles loading and initial cleaning of raw data.
  - `preprocessing.py`: Contains functions for feature engineering and creating sequential windows.
  - `models.py`: Defines the neural network architectures (LSTM and Transformer-based DQNs).
  - `trainer.py`: Implements the core RL training loop and evaluation logic.
  - `utils.py`: Provides utility functions like saving results and setting random seeds.
- `scripts/`: Entry point scripts to run the project.
  - `train.py`: Runs the main RL training phase on the entire dataset.
  - `train_clustered.py`: Runs the clustering-based training workflow.
- `requirements.txt`: A list of required Python packages.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alpapanau/RLFD.git
    cd RLFD
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the models directory:**
    The scripts will save models to a `models/` directory. Create it first:
    ```bash
    mkdir models
    ```

5.  **Place your data:**
    Place your raw CSV files in a directory (e.g., `data/bonifico/`) and ensure the `data.path` key in `config/params.yaml` points to this directory.

## How to Run the Workflows

This project supports two distinct training methodologies. 

### Workflow 1: Standard End-to-End Training

This workflow trains a single, global agent on the entire dataset.

1.  **Configure:** Open `config/params.yaml` and adjust parameters under the `training`, `model`, and `preprocessing` sections.
2.  **Run Training:**
    ```bash
    python scripts/train.py
    ```
    This script trains the agent, saves the best-performing model state to the path specified in `data.model_save_path`, and logs the final and best model results to the results CSV.

### Workflow 2: Clustered Training

This alternative workflow first clusters transactions into behavioral groups and then trains a specialized agent for each cluster. 

1.  **Configure:** Open `config/params.yaml` and adjust the parameters under the `clustering_training` section. You can select which features to use for clustering and tune the DBSCAN algorithm's hyperparameters (`eps`, `min_samples`).
2.  **Run Clustered Training:**
    ```bash
    python scripts/train_clustered.py
    ```
    This script will:
    - Cluster the entire dataset using DBSCAN.
    - For each cluster, initialize and train a dedicated RL agent on all clients belonging to that cluster.
    - Evaluate each agent on its cluster's test data.
    - Aggregate the predictions from all cluster agents and compute global performance scores.
    - The results are logged to the CSV file with `model_version` tags like `clustered_training_best` and `clustered_training_final` to distinguish them.