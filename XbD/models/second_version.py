# Here version with transformer over the N detections with a final a CLS token
# Then, the final classification is done with a linear layer over the CLS token.

# This is a new idea for the XbD model, which uses a transformer to process the sequence of detections.

import torch
import torch.nn as nn
import torch.nn.functional as F

class XbD_SecondVersion(nn.Module):
    def __init__(self,
                 N: int,
                 num_classes: int = 41,
                 d_model: int = 32,
                 nhead: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.N = N

        # 1) project num_classes→d_model
        self.up_proj = nn.Linear(num_classes, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        # 2) learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) Transformer w/ dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,         # dropout in attention & FFN
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) classifier on CLS + dropout
        self.cls_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        assert N == self.N

        # — up‐project each detection
        x = x.view(B * T * N, C)
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.proj_norm(x)
        x = self.proj_drop(x)
        x = x.view(B, T, N, self.d_model)

        # — flatten B, T to feed transformer
        x = x.view(B * T, N, self.d_model)

        # — prepend CLS
        cls = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls, x], dim=1)  # → (B*T, N+1, d_model)

        # — transformer
        x = self.transformer(x)         # → (B*T, N+1, d_model)

        # — extract CLS, dropout, classify
        cls_out = x[:, 0, :]            # → (B*T, d_model)
        cls_out = self.cls_drop(cls_out)
        logits  = self.classifier(cls_out)  # → (B*T, 7)

        return logits.view(B, T, -1)