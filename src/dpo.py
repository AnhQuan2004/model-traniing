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
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from huggingface_hub import login
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import DataCollatorForLanguageModeling

from utils import load_yaml_config, load_and_merge_datasets, format_dpo_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def dpo_pipeline(config_path: str):
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
        project_name="qwen3-4b-medical-dpo",
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

    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = "qwen3",
    # )

    dataset = dataset.map(
        format_dpo_dataset,
        fn_kwargs = {"tokenizer": tokenizer},
        num_proc = 4,
        remove_columns = dataset.column_names,
    )

    dataset = dataset.train_test_split(
        test_size=0.1,
        seed=3047,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # DATA COLLATOR
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer = tokenizer,
    #     mlm = False,
    # )

    trainer = DPOTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        # data_collator = data_collator,
        # packing = True,
        beta = 0.1,
        ref_model = None,  # No reference model for DPO
        
        # TRAINING ARGUMENTS CONFIGS
        args = DPOConfig(
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
    
            max_length = max_seq_length,
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
    dpo_pipeline(args.config_path)
