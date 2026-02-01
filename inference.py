# inference.py
from transformers import pipeline
from peft import PeftModel, PeftConfig

peft_model_id = "./whisper-large-v2-lora/final"  # or huggingface repo id

config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    config.base_model_name_or_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)

processor = WhisperProcessor.from_pretrained(config.base_model_name_or_path)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16
)

# Example
audio_path = "your_audio.mp3"
result = pipe(audio_path)
print(result["text"])
