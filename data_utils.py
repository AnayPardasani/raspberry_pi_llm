# data_utils.py
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def get_dataset(dataset_name="mozilla-foundation/common_voice_11_0", 
                lang="hi", 
                cache_dir=None):
    common_voice = DatasetDict()
    
    common_voice["train"] = load_dataset(
        dataset_name, lang, split="train", cache_dir=cache_dir
    )
    common_voice["test"] = load_dataset(
        dataset_name, lang, split="test", cache_dir=cache_dir
    )
    
    # Keep only relevant columns
    common_voice = common_voice.remove_columns([
        "accent", "age", "client_id", "down_votes", 
        "gender", "locale", "path", "segment", "up_votes"
    ])
    
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    return common_voice
