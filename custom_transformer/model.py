# model.py - Keyword Transformer style (KWT-1/2/3 variant)
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(40, 98), patch_size=(40, 1), embed_dim=512):
        super().__init__()
        self.num_patches = (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, 1, H, W) → typically (B, mel_bins, time_frames)
        x = x.unsqueeze(1)  # add channel
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class KeywordTransformer(nn.Module):
    def __init__(self, num_classes=35, img_size=(40, 98), patch_size=(40, 1),
                 embed_dim=512, depth=12, nhead=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, nhead, embed_dim * mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.dropout(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x.mean(dim=1))  # global average pool over patches
        return self.head(x)
