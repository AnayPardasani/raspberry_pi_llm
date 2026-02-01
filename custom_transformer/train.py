# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from dataset import SpeechCommandsDataset
from model import KeywordTransformer

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = "/kaggle/input/speech-commands"  # adjust path
    
    mel_transform = MelSpectrogram(
        sample_rate=16000,
        n_mels=40,
        n_fft=480,
        hop_length=160,
        win_length=480,
        window_fn=torch.hamming_window,
        power=2.0
    ).to(DEVICE)
    
    train_ds = SpeechCommandsDataset(ROOT_DIR, split='train', transform=mel_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    # Example: 35 classes for v0.02 full
    model = KeywordTransformer(num_classes=35, depth=6, embed_dim=256, nhead=4).to(DEVICE)  # smaller for Kaggle
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for specs, labels in tqdm(train_loader):
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f"Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f} | Acc {100.*correct/total:.2f}%")
        scheduler.step()
    
    torch.save(model.state_dict(), "transformer_white_v2.pth")

if __name__ == "__main__":
    main()
