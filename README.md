# Fraud Detection using Reinforcement Learning (RLFD)

This project implements a Reinforcement Learning-based approach for detecting anomalies in sequential data, such as financial transactions. It uses a Deep Q-Network (DQN) with an LSTM or Transformer backend to learn a policy for classifying transactions as normal or fraudulent.

The project features a two-stage workflow:
1.  **Main Training:** A robust Reinforcement Learning agent is trained on the full dataset. The best-performing model based on validation metrics is saved.
2.  **Targeted Fine-Tuning:** The saved model is reloaded and fine-tuned on a small, curated set of the most "uncertain" examples, allowing it to specialize in difficult edge cases.

## Project Structure

- `config/`: Contains the central `params.yaml` configuration file for managing all hyperparameters.
- `data/`: Placeholder for the input datasets. See `data/README.md` for details on where to place your data.
- `models/`: Default directory for saving and loading trained model states (`.pth` files).
- `rlfd/`: The main Python source code package.
  - `data_loader.py`: Handles loading and initial cleaning of raw data.
  - `preprocessing.py`: Contains functions for feature engineering and creating sequential windows.
  - `models.py`: Defines the neural network architectures (LSTM and Transformer-based DQNs).
  - `trainer.py`: Implements the core RL training loop and evaluation logic.
  - `utils.py`: Provides utility functions like saving results and setting random seeds.
- `scripts/`: Entry point scripts to run the project.
  - `train.py`: Runs the main RL training phase and saves the best model.
  - `fine_tune.py`: Loads a pre-trained model and performs targeted fine-tuning on uncertain samples.
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
    Place your raw CSV files in a directory (e.g., `Bancari/Data/bonifico/`) and ensure the `data.path` key in `config/params.yaml` points to this directory.

## How to Run the Workflow

The project is designed as a two-step process.

### Step 1: Main Model Training

First, train the base model using the RL approach. This script will evaluate the model on a validation set during training and save the state of the best-performing model.

1.  **Configure:** Open `config/params.yaml` and adjust the parameters under the `data`, `preprocessing`, `model`, and `training` sections. Ensure `data.model_save_path` is set to where you want the best model saved.
2.  **Run Training:**
    ```bash
    python scripts/train.py
    ```
    This will create a model file (e.g., `models/best_model.pth`). Results from this run will be logged to the CSV file specified in the config, marked with `model_version: best_validation`.

### Step 2: Targeted Fine-Tuning (Optional)

After a base model has been trained and saved, you can run this script to fine-tune it on the examples it found most difficult.

1.  **Configure:** Open `config/params.yaml`.
    -   Ensure the `model_save_path` points to the model you just trained.
    -   Enable the fine-tuning phase by setting `fine_tuning.enabled: true`.
    -   Adjust the fine-tuning hyperparameters (`k_percent_uncertain`, `epochs`, `learning_rate`) to your needs.
2.  **Run Fine-Tuning:**
    ```bash
    python scripts/fine_tune.py
    ```
    This script will:
    - Load the model from `model_save_path`.
    - Identify the most uncertain training samples.
    - Fine-tune the model on these samples.
    - Evaluate the newly fine-tuned model on the test set.
    - Save the final, fine-tuned model to the path specified by `fine_tuned_model_save_path`.
    - Log the results, marked with `model_version: fine_tuned`.

By comparing the `best_validation` and `fine_tuned` results in your log file, you can directly measure the impact of the targeted fine-tuning process.