import yaml
import re
import json
from loguru import logger
import datasets
from datasets import load_dataset, concatenate_datasets

def load_yaml_config(path: str) -> dict:
    """Load a yaml file and return a dictionary."""

    try: 
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading yaml file {path}: {e}")
        raise e

def formatting_prompts_func(examples, tokenizer):
    return {"text" : [example + tokenizer.eos_token for example in examples["text"]]}


def load_and_merge_datasets(config: dict) -> datasets.Dataset | datasets.DatasetDict:
    """Load and merge datasets."""

    dataset_config = config.get("datasets", {})
    dataset_names = dataset_config.get("names", [])
    
    # If splits are defined in the config, load the dataset with all its splits.
    # This typically returns a DatasetDict. We assume only one dataset is used in this case.
    if "splits" in dataset_config:
        if len(dataset_names) > 1:
            logger.warning("Multiple datasets found with splits config. Only the first one will be used.")
        return load_dataset(dataset_names[0])

    # Original behavior: load only the train split and concatenate
    loaded_datasets = []
    for name in dataset_names:
        loaded_datasets.append(load_dataset(name, split="train"))

    return concatenate_datasets(loaded_datasets).shuffle(seed=3047)

def apply_chat_template(example, tokenizer):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Your goal is to provide accurate and safe information to the user."
        },
        {
            "role": "user",
            "content": str(example['instruction']) + ('\n' + example["input"] if "input" in example and example["input"] else "")},
        {
            "role": "assistant",
            "content": example.get("output", "")}
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        'text': chat_format
    }

def format_dpo_dataset(example, tokenizer):
    rejected_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example["rejected"]}
    ]

    chosen_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example['chosen']}
    ]
    
    return {
        'rejected': tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        'chosen': tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    }

def format_goemotions_for_chat(example, tokenizer):
    # Find the emotion label(s)
    emotions = []
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    for col in emotion_columns:
        if col in example and example[col] == 1:
            emotions.append(col)

    output_text = ", ".join(emotions) if emotions else "neutral"

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specializing in emotion classification. Your task is to identify the emotions present in the user's text."
        },
        {
            "role": "user",
            "content": f"Please classify the emotion of the following text: {example['text']}"
        },
        {
            "role": "assistant",
            "content": output_text
        }
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': chat_format}
