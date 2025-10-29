import os
import comet_ml
from comet_ml import Experiment
import argparse
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
import torch

import sys
import typing
sys.modules['typing'].Any = typing.Any

import builtins
builtins.Any = typing.Any

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from huggingface_hub import login
import datasets
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForLanguageModeling

from utils import load_yaml_config, load_and_merge_datasets, apply_chat_template, format_goemotions_for_chat

def sft_pipeline(config_path: str):
    """Finetune the model."""

    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    load_dotenv()
    # LOAD CONFIG
    config = load_yaml_config(config_path)
    os.environ["COMET_LOG_ASSETS"] = "True"
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="qwen3-4b-medical-cpt",
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

    dataset = load_and_merge_datasets(config)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3",
    )

    # Determine which formatting function to use based on the dataset
    dataset_name = config["datasets"]["names"][0]
    if "go_emotions" in dataset_name:
        formatting_function = format_goemotions_for_chat
    else:
        formatting_function = apply_chat_template

    # Determine the columns to remove from one of the splits (e.g., 'train')
    # This is necessary because a DatasetDict doesn't have a single `column_names` attribute.
    if isinstance(dataset, datasets.DatasetDict):
        column_names = dataset['train'].column_names
    else:
        column_names = dataset.column_names

    dataset = dataset.map(
        formatting_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=column_names
    )

    # Check if the loaded dataset is a DatasetDict and contains the specified splits
    if isinstance(dataset, datasets.DatasetDict):
        splits_config = config.get("datasets", {}).get("splits", {})
        train_split = splits_config.get("train", "train")
        eval_split = splits_config.get("validation", "validation")

        if train_split in dataset and eval_split in dataset:
            train_dataset = dataset[train_split]
            eval_dataset = dataset[eval_split]
            logger.info(f"Using '{train_split}' for training and '{eval_split}' for evaluation.")
        else:
            logger.warning("Specified splits not found, falling back to random split.")
            dataset = dataset.train_test_split(test_size=0.1, seed=3047)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
    else:
        # Fallback for single datasets
        dataset = dataset.train_test_split(test_size=0.1, seed=3047)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    
    # DATA COLLATOR
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
        # packing = True,
        
        # TRAINING ARGUMENTS CONFIGS
        args = SFTConfig(
            per_device_train_batch_size = config["training_args"]["per_device_train_batch_size"],
            per_device_eval_batch_size = config["training_args"]["per_device_eval_batch_size"],
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
            seed = config["training_args"]["seed"],
            output_dir = config["training_args"]["output_dir"],
            report_to = config["training_args"]["report_to"],
    
            max_seq_length = max_seq_length,
            dataset_num_proc = config["datasets"]["preprocessing"]["num_proc"]
        )
    )

    trainer = trainer.train()

    model.push_to_hub(config["artifacts"]["model_hub_id"])
    tokenizer.push_to_hub(config["artifacts"]["model_hub_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    sft_pipeline(args.config_path)
