# src/dpo.py
import os
import argparse
import sys
import typing
import builtins

# --- Unsloth phải import TRƯỚC transformers/hf ---
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
# -------------------------------------------------

import comet_ml
from comet_ml import Experiment
from loguru import logger
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import DatasetDict
from trl import DPOTrainer, DPOConfig

from utils import (
    load_yaml_config,
    load_and_merge_datasets,
    format_dpo_dataset,  # phải trả ra các cột: prompt, chosen, rejected
)

# typing compatibility (chỉ khi env cần)
sys.modules["typing"].Any = typing.Any
builtins.Any = typing.Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # tuỳ chọn

def _pick_dtype_str(cfg_val: typing.Optional[str]) -> str:
    """
    Trả về dtype string cho Unsloth: 'bfloat16' | 'float16' | 'float32'.
    Ưu tiên cấu hình YAML nếu có, không thì auto theo phần cứng.
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


def dpo_pipeline(config_path: str):
    """Direct Preference Optimization fine-tuning (Unsloth + TRL)."""
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    load_dotenv()

    # -------------------------
    # LOAD CONFIG + COMET INIT
    # -------------------------
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

    # -------------------------
    # MODEL ARGS
    # -------------------------
    model_name = str(config["model_args"]["name"])
    max_seq_length = int(config["model_args"]["max_seq_length"])
    yaml_dtype = config["model_args"].get("dtype")
    load_in_4bit = bool(config["model_args"]["load_in_4bit"])
    dtype_str = _pick_dtype_str(yaml_dtype)

    # -----------------------------------
    # LOAD BASE MODEL (Unsloth-native)
    #   - KHÔNG truyền config=... hay torch_dtype
    #   - Dùng dtype string + max_seq_length
    # -----------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype_str,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )

    # -----------------------------------
    # DATASET (DPO: cần prompt/chosen/rejected)
    # -----------------------------------
    dataset = load_and_merge_datasets(config)

    # Dùng chat template Qwen3 trước khi format DPO (nếu format cần tokenizer)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3",
    )

    num_proc = int(config["datasets"]["preprocessing"].get("num_proc", 1))
    dataset = dataset.map(
        format_dpo_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    # Bảo đảm có split train/test
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        dataset = dataset.train_test_split(test_size=0.1, seed=int(config["training_args"].get("seed", 3407)))
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)

    # -----------------------------------
    # DPO TRAINER
    # -----------------------------------
    dpo_args = DPOConfig(
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

        # ===== mixed precision =====
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # ===== sequence lengths =====
        # Unsloth DPOConfig KHÔNG nhận max_target_length / max_prompt_length
        max_length=int(max_seq_length),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        reference_free=True,   # Không dùng ref model -> đặt cờ này
        # ref_model=None,      # không cần truyền khi reference_free=True
    )

    trainer.train()

    # -----------------------------------
    # PUSH ARTIFACTS
    # -----------------------------------
    hub_id = str(config["artifacts"]["model_hub_id"])
    try:
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        logger.success(f"Pushed DPO adapters/tokenizer to HF Hub: {hub_id}")
    except Exception as e:
        logger.error(f"Push to hub failed: {e}")

    # log last metrics to Comet (nếu có)
    if trainer.state.log_history:
        last = {k: v for k, v in trainer.state.log_history[-1].items() if isinstance(v, (int, float))}
        for k, v in last.items():
            experiment.log_metric(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    dpo_pipeline(args.config_path)