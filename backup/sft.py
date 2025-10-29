# src/sft.py
import os
import argparse
import sys
import typing
import builtins

# --- Unsloth MUST be imported before transformers/hf bits ---
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
# ------------------------------------------------------------

import comet_ml
from comet_ml import Experiment
from loguru import logger
from dotenv import load_dotenv

import torch
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

from utils import load_yaml_config, load_and_merge_datasets, apply_chat_template

# typing compatibility hacks (only if your env needs them)
sys.modules["typing"].Any = typing.Any
builtins.Any = typing.Any


def _pick_dtype_str(cfg_val: typing.Optional[str]) -> str:
    """
    Return a dtype *string* preferred by Unsloth: "bfloat16" | "float16" | "float32".
    If cfg_val is provided, honor it; else auto-detect via is_bfloat16_supported().
    """
    if cfg_val:
        v = str(cfg_val).lower()
        if v in ("bf16", "bfloat16"):
            return "bfloat16"
        if v in ("fp16", "float16", "half"):
            return "float16"
        if v in ("fp32", "float32"):
            return "float32"
    return "bfloat16" if is_bfloat16_supported() else "float16"


def sft_pipeline(config_path: str):
    """Finetune the model with LoRA using Unsloth/TRL."""
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    load_dotenv()

    # -------------------------
    # LOAD CONFIG + COMET INIT
    # -------------------------
    config = load_yaml_config(config_path)
    os.environ["COMET_LOG_ASSETS"] = "True"
    experiment = Experiment(  # keep a handle if you want to log params/metrics
        api_key=os.getenv("COMET_API_KEY"),
        project_name="qwen3-4b-medical-sft",
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set.")
        raise ValueError("HF_TOKEN is not set.")
    login(token=HF_TOKEN)

    # -------------------------
    # MODEL ARGS
    # -------------------------
    model_name = config["model_args"]["name"]
    max_seq_length = int(config["model_args"]["max_seq_length"])
    yaml_dtype = config["model_args"].get("dtype")
    load_in_4bit = bool(config["model_args"]["load_in_4bit"])
    dtype_str = _pick_dtype_str(yaml_dtype)

    # -----------------------------------
    # LOAD BASE MODEL (Unsloth-native)
    #   - use max_seq_length (NOT max_length)
    #   - pass dtype as string for Unsloth
    #   - DO NOT pass config=... or torch_dtype
    # -----------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype_str,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,  # Unsloth handles remote code safely
    )

    # -----------------------------------
    # APPLY LoRA
    # -----------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        **config["lora_args"],
    )

    # -----------------------------------
    # DATASET
    # -----------------------------------
    dataset = load_and_merge_datasets(config)

    # Qwen3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3",
    )

    num_proc = int(config["datasets"]["preprocessing"].get("num_proc", 1))
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    # simple split (if your dataset already has splits, adapt here)
    dataset = dataset.train_test_split(
        test_size=0.1,
        seed=int(config["training_args"].get("seed", 3407)),
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -----------------------------------
    # TRAINER
    # -----------------------------------
    # Let Unsloth pick kernels based on dtype; no need to force fp16/bf16 flags.
    sft_args = SFTConfig(
        # ===== core =====
        per_device_train_batch_size=int(config["training_args"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training_args"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["training_args"]["gradient_accumulation_steps"]),
        num_train_epochs=float(config["training_args"]["num_train_epochs"]),
        warmup_ratio=float(config["training_args"]["warmup_ratio"]),
        learning_rate=float(config["training_args"]["learning_rate"]),
        weight_decay=float(config["training_args"]["weight_decay"]),
        logging_steps=int(config["training_args"]["logging_steps"]),
        optim=str(config["training_args"]["optim"]),
        lr_scheduler_type=str(config["training_args"]["lr_scheduler_type"]),
        seed=int(config["training_args"]["seed"]),
        output_dir=str(config["training_args"]["output_dir"]),
        report_to=list(config["training_args"]["report_to"]),

        # ===== saving =====
        save_steps=int(config["training_args"].get("save_steps", 500)),
        save_total_limit=int(config["training_args"].get("save_total_limit", 5)),

    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=sft_args,
        # packing=True,  # enable if desired and your data is unformatted
    )

    trainer.train()

    # -----------------------------------
    # PUSH ARTIFACTS (LoRA adapters + tokenizer)
    # -----------------------------------
    hub_id = str(config["artifacts"]["model_hub_id"])
    try:
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        logger.success(f"Pushed LoRA adapters + tokenizer to HF Hub: {hub_id}")
    except Exception as e:
        logger.error(f"Push to hub failed: {e}")

    # optional: log final metrics to Comet
    if trainer.state.log_history:
        last = {k: v for k, v in trainer.state.log_history[-1].items() if isinstance(v, (int, float))}
        for k, v in last.items():
            experiment.log_metric(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    sft_pipeline(args.config_path)