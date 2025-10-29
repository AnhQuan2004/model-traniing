import os 
import comet_ml
from comet_ml import Experiment
import argparse
from loguru import logger
from dotenv import load_dotenv
import torch

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from huggingface_hub import login
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForLanguageModeling

from utils import load_yaml_config

def training_pipeline(config_path: str):
    """Training pipeline for continual pretraining."""

    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    load_dotenv()
    # Load config
    config = load_yaml_config(config_path)
    os.environ["COMET_LOG_ASSETS"] = "True"
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="qwen3-1.7b-cpt",
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set.")
        raise ValueError("HF_TOKEN is not set.")

    login(token=HF_TOKEN)
    
    # MODEL CONFIGS
    model_name = config["model_args"]["name"]
    max_seq_length = config["model_args"]["max_seq_length"]
    dtype = config["model_args"]["dtype"]
    load_in_4bit = config["model_args"]["load_in_4bit"]
    full_finetuning = config["model_args"]["full_finetuning"]

    # MODEL & TOKENIZER
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = full_finetuning,
    )

    train_dataset = load_dataset(
        config["datasets"]["sources"],
        split = config["datasets"]["splits"],
    )

    

    # DATA COLLATOR
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        data_collator = data_collator,
        
        # TRAINING ARGUMENTS CONFIGS
        args = SFTConfig(
            per_device_train_batch_size = config["training_args"]["per_device_train_batch_size"],
            gradient_accumulation_steps = config["training_args"]["gradient_accumulation_steps"],
            num_train_epochs = config["training_args"]["num_train_epochs"],
            warmup_ratio = config["training_args"]["warmup_ratio"],

            learning_rate = float(config["training_args"]["learning_rate"]),
            weight_decay = float(config["training_args"]["weight_decay"]),
            logging_steps = config["training_args"]["logging_steps"],

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),

            optim = config["training_args"]["optim"],
            lr_scheduler_type = config["training_args"]["lr_scheduler_type"],
            lr_scheduler_kwargs = {"min_lr_rate": 0.1} if config["training_args"]["lr_scheduler_type"] == "cosine_with_min_lr" else None,
            seed = config["training_args"]["seed"],
            output_dir = config["training_args"]["output_dir"],
            report_to = config["training_args"]["report_to"],
            
            max_seq_length = max_seq_length,
            dataset_num_proc = config["datasets"]["preprocessing"]["num_proc"],
            packing = True,
        )
    )

    trainer = trainer.train()

    model.push_to_hub(config["artifacts"]["model_hub_id"])
    tokenizer.push_to_hub(config["artifacts"]["model_hub_id"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    training_pipeline(args.config_path)