# Here, we add the temporal dimension to the model, so that it can process sequences of frames.
# A temporal transformer is used to create the context between [CLS] tokens of each frame.

import torch
import torch.nn as nn
import torch.nn.functional as F

class XbD_ThirdVersion(nn.Module):
    def __init__(self,
                 N: int,
                 num_classes: int = 41,
                 d_model: int = 64,
                 nhead_det: int = 8,
                 num_layers_det: int = 2,
                 nhead_time: int = 8,
                 num_layers_time: int = 2,
                 dropout: float = 0.1,):
        super().__init__()
        self.N = N
        self.d_model = d_model

        # 1) project each detection (size=num_classes) → d_model
        self.up_proj    = nn.Linear(num_classes, d_model)
        self.proj_norm  = nn.LayerNorm(d_model)
        self.proj_drop  = nn.Dropout(dropout)

        # 2) a learnable [CLS] token for detection‐level transformer
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) detection‐level transformer (over N+1 tokens: cls + detections)
        det_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_det,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_det = nn.TransformerEncoder(
            det_layer,
            num_layers=num_layers_det
        )

        # 4) time‐level transformer (over T cls‐embeddings)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_time,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_time = nn.TransformerEncoder(
            time_layer,
            num_layers=num_layers_time
        )

        # 5) Dropout on CLS before final classification
        self.cls_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, N, num_det_features)
        Returns:
            logits: Tensor of shape (B, T, num_ego_classes)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got N={N}"

        # — Project & regularize
        x = x.view(B * T * N, C)
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.proj_norm(x)
        x = self.proj_drop(x)
        x = x.view(B, T, N, self.d_model)

        # — Detection‐level transformer
        x = x.view(B * T, N, self.d_model)
        cls_tokens = self.cls_token.expand(B * T, -1, -1)          # (B*T, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                      # (B*T, N+1, d_model)
        x = self.transformer_det(x)                                # (B*T, N+1, d_model)
        cls_det = x[:, 0, :]                                       # (B*T, d_model)
        cls_det = cls_det.view(B, T, self.d_model)                 # (B, T, d_model)

        # — Time‐level transformer
        #    (B, T, d_model) → (B, T, d_model)
        ctx_cls = self.transformer_time(cls_det)

        # — Classify each time‐step with dropout
        ctx_cls = self.cls_drop(ctx_cls)
        logits  = self.classifier(ctx_cls)                         # (B, T, 7)
        return logits