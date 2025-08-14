import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim=64*16, embedded_dim=256, patch_size=(8,4)):
        super(PatchEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedded_dim = embedded_dim
        self.patch_size = patch_size

        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embedded_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        """
        X.size = (B, T, H, W) = (B, 101, 64, 16)
        """
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        x = self.projection(x)      # (B*T, embedded_dim, 8, 4)
        x = x.flatten(2, 3)         # (B*T, embedded_dim, 8*4)
        x = x.transpose(1, 2)   # (B*T, 32, embedded_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self):
        """
        Initialize Positional embeddings (and a cls token) in the shape (1, num_patches+1, embedded_dim)
        """
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, 33, 256))
        self.cls = nn.Parameter(torch.zeros(1, 1, 256))
    def forward(self, x):
        """
        Expand [CLS] token to shape [B, 1, 256]
        Concatenate [CLS] and patch embeddings → [B, 33, 256]
        Add pos_embed of shape [1, 33, 256]
        Return the result
        """
        # x: (B*T, 32, 256)
        B = x.shape[0]
        cls_token = self.cls.expand(B, -1, -1)      # cls: (B, 1, 256)
        # print(f"cls_token shape: {cls_token.shape}, x shape: {x.shape}")
        x = torch.cat((cls_token, x), dim=1)       # x: (B*T, 33, 256)
        x = x + self.position_embedding              # x: (B*T, 33, 256)
        return x

class transfer_ViT(nn.Module):
    def __init__(self, output_features):
        super(transfer_ViT, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64*16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)
        )
        self.patch_embedding = PatchEmbedding()
        self.positional_encoding = PositionalEncoding()
        self.output_features = output_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc_out = nn.Linear(256, output_features)
    def forward(self, x):
        x = x.reshape(-1, 100, 64*16)
        x = self.mlp(x)
        x = x.reshape(-1, 100, 64, 16)
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)     # x: (B*T, 33, 256)
        x = self.transformer_encoder(x)     # x: (B*T, 33, 256)
        x = self.fc_out(x[:, 0, :])     # only use cls's output as output
        x = x.view(-1, 100, self.output_features)  # output (grf) (B, 101, output_features)
        return x