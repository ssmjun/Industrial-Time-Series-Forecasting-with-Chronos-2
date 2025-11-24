# Industrial Time Series Forecasting with Chronos-2

This repository contains code for forecasting industrial time series data using the Chronos-2 model.

## 1. Installation

### A. Set up Conda Environment
1. Install Conda if not already installed ([Instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).

2. Create and activate a new environment (Python 3.10 recommended):
    ```bash
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```

### B. Install Dependencies
**Important:** Please install PyTorch with CUDA support first to enable GPU acceleration.

1. Install PyTorch (Visit [pytorch.org](https://pytorch.org/get-started/locally/) for your specific CUDA version):
    ```bash
    # Example for Linux/Windows with CUDA 12.6 (Adjust as needed)
    pip install torch --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
    ```

2. Install other requirements:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Usage

You can run the full suite of experiments (Baseline to Fine-tuning) using the provided shell script.

1. Grant execution permission (first time only):
    ```bash
    chmod +x run_experiments.sh
    ```

2. Run experiments:
    ```bash
    ./run_experiments.sh
    ```
    *Or run manually:* `bash run_experiments.sh`

### Experiment Scenarios included:
- **Baseline:** Linear Regression & Chronos (Zero-shot)
- **Covariates:** Using process data (e.g., CO2, Temp) as context
- **Cross-Learning:** Multivariate forecasting
- **Fine-tuning:** Training Chronos on Samyang data
