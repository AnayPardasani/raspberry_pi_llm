# train.py
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_utils import get_dataset, prepare_dataset
from model_utils import load_model_and_processor, apply_lora
import torch

def main():
    MODEL_NAME = "openai/whisper-large-v2"
    DATA_LANG = "hi"               # change to your language code
    OUTPUT_DIR = "./whisper-large-v2-lora"
    
    # Load data
    dataset = get_dataset(lang=DATA_LANG)
    
    model, processor = load_model_and_processor(MODEL_NAME, use_4bit=True)
    model = apply_lora(model)
    
    # Prepare datasets
    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=4  # adjust based on CPU cores
    )
    
    # Training arguments (typical values from such notebooks)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,          # adjust based on GPU
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,                         # or use num_train_epochs
        gradient_checkpointing=True,
        fp16=True,                              # or bf16=True
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",                       # can be "tensorboard" / "wandb"
    )
    
    # You would normally define a compute_metrics function with jiwer here
    from evaluate import load
    wer_metric = load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=lambda data: {
            "input_features": torch.stack([f["input_features"] for f in data]),
            "labels": processor.tokenizer.pad(
                [f["labels"] for f in data], 
                return_tensors="pt"
            ).input_ids
        },
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # trick to save preprocessor
    )
    
    trainer.train()
    
    # Save final LoRA adapter + merge if desired
    trainer.save_model(OUTPUT_DIR + "/final")
    processor.save_pretrained(OUTPUT_DIR + "/final")

if __name__ == "__main__":
    main()
