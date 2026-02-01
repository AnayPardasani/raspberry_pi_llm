# inference.py
import torch
from torchaudio import load
from model import KeywordTransformer
from dataset import mel_transform  # or redefine

model = KeywordTransformer(num_classes=35)
model.load_state_dict(torch.load("transformer_white_v2.pth"))
model.eval()

waveform, sr = load("test_command.wav")
spec = mel_transform(waveform)
with torch.no_grad():
    logits = model(spec.unsqueeze(0))
    pred = logits.argmax(-1).item()
print(f"Predicted class: {pred}")
