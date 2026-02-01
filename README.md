# Whisper Large-v2 Fine-Tuning with LoRA (PEFT)

Efficient fine-tuning of OpenAI's **Whisper-large-v2** using **LoRA** (Low-Rank Adaptation) via Hugging Face's **PEFT** library. This approach drastically reduces VRAM usage and storage requirements while maintaining strong performance — enabling training on consumer GPUs (e.g., T4, RTX 3090/4090) or free Kaggle/Colab sessions.

Originally inspired by community notebooks for parameter-efficient ASR fine-tuning.

## Features

- 4-bit quantization (bitsandbytes) + gradient checkpointing for ~7–10 GB VRAM usage
- LoRA adapters applied only to key attention modules (`q_proj`, `v_proj`)
- Full Seq2SeqTrainer integration with Transformers
- Automatic mixed precision (fp16/bf16)
- Easy inference with PEFT + pipeline API
- Modular file structure for better maintainability and reproducibility

## Project Structure
