
# Fraud Detection using Reinforcement Learning (RLFD)

This project implements the RLFD architecture for detecting anomalies in sequential data, such as financial transactions. It uses a Deep Q-Network (DQN) with an LSTM or Transformer backend to learn a policy for classifying transactions as normal or fraudulent.

## Project Structure

- `config/`: Contains configuration files (YAML) for managing all hyperparameters.
- `data/`: Placeholder for the input datasets. See `data/README.md` for details.
- `rlfd/`: The main source code package.
  - `data_loader.py`: Handles loading and initial cleaning of raw data.
  - `preprocessing.py`: Contains functions for feature engineering and creating sequential windows.
  - `models.py`: Defines the neural network architectures (LSTM and Transformer-based DQNs).
  - `trainer.py`: Implements the core RL training loop, evaluation, and model saving.
  - `utils.py`: Provides utility functions like saving results and setting random seeds.
- `scripts/`: Entry point scripts to run the project.
  - `train.py`: The main script to start a training and evaluation run.
- `requirements.txt`: A list of required Python packages.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rlfd_anomaly_detection
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

4.  **Place your data:**
    Place your raw CSV files in a directory and update the `data_path` in `config/params.yaml`.

## How to Run

1.  **Configure your experiment:**
    Adjust the hyperparameters in `config/params.yaml` to fit your needs. You can control data paths, preprocessing steps, model architecture, and training parameters.

2.  **Run the training script:**
    ```bash
    python scripts/train.py
    ```

The script will load the data, preprocess it, train the model, evaluate it on a test set, and save the results to a CSV file as specified in the configuration.