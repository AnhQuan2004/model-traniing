# Language Model Fine-Tuning Repository

This repository provides a comprehensive suite of tools for fine-tuning and aligning language models. It supports Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Continual Pre-Training (CPT), allowing you to adapt pre-trained models for a wide range of applications.

## Features

- **Supervised Fine-Tuning (SFT):** Fine-tune models for specific tasks using labeled datasets.
- **Direct Preference Optimization (DPO):** Align models with human preferences by training on comparative data.
- **Continual Pre-Training (CPT):** Expand your model's knowledge base with new data before fine-tuning.
- **Highly Configurable:** Easily customize training runs through YAML configuration files.
- **Hugging Face Integration:** Seamlessly load models and datasets from the Hugging Face Hub.
- **Experiment Tracking:** Monitor and analyze training metrics with Comet ML.

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### 2. Configuration

All training parameters are managed through YAML files in the `configs` directory. Before running a script, you can customize the following:

- **Base Model:** Specify the pre-trained model to fine-tune.
- **Datasets:** Define the datasets for training, validation, and testing.
- **Training Arguments:** Adjust hyperparameters like learning rate, batch size, and number of epochs.

### 3. Running the Training Scripts

To run a training script, use the following command format:

```bash
python src/<script_name>.py --config_path configs/<config_file>.yaml
```

**Example Scripts:**

- **Supervised Fine-Tuning (SFT):**
  ```bash
  python src/sft.py --config_path configs/qwen_sft_config.yaml
  ```

- **Direct Preference Optimization (DPO):**
  ```bash
  python src/dpo.py --config_path configs/qwen_dpo_config.yaml
  ```

- **Continual Pre-Training (CPT):**
  ```bash
  python src/continual_pretraining.py --config_path configs/qwen_cpt_config.yaml
  ```

## Experiment Tracking

This project is integrated with Comet ML for experiment tracking. To use this feature, make sure to set your Comet ML API key as an environment variable.
