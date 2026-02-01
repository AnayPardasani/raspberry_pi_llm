# model_utils.py
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def load_model_and_processor(model_name="openai/whisper-large-v2", 
                             use_4bit=True):
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.feature_extractor.return_attention_mask = True  # important for training
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_safetensors=True
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return model, processor

def apply_lora(model, r=32, lora_alpha=64, lora_dropout=0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"  # Whisper uses causal LM head
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model
