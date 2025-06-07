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
                 num_layers_time: int = 2,):
        """
        Args:
            N:                 number of detections per time step
            num_classes:       dimensionality of each detection (41), number of concepts
            d_model:           embedding dimension
            nhead_det:         attention heads for detection‐level transformer
            num_layers_det:    layers in detection‐level transformer
            nhead_time:        attention heads for time‐level transformer
            num_layers_time:   layers in time‐level transformer
        """
        super().__init__()
        self.N = N
        self.d_model = d_model

        # 1) project each detection (size=num_classes) → d_model
        self.up_proj = nn.Linear(num_classes, d_model)

        # 2) a learnable [CLS] token for detection‐level transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) detection‐level transformer (over N+1 tokens: cls + detections)
        det_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_det,
            dim_feedforward=d_model * 4,
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
            batch_first=True
        )
        self.transformer_time = nn.TransformerEncoder(
            time_layer,
            num_layers=num_layers_time
        )

        # 5) final classifier on each time‐step’s contextualized [CLS]
        self.classifier = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, N, 41)
        Returns:
            logits: Tensor of shape (B, T, num_ego_classes)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got N={N}"
        # project → (B*T*N, d_model)
        x = x.view(B * T * N, C)
        x = self.up_proj(x)
        x = F.relu(x)
        #    → (B, T, N, d_model)
        x = x.view(B, T, N, self.d_model)

        # detection‐level transformer per time‐step:
        # flatten batch & time: (B*T, N, d_model)
        x = x.view(B * T, N, self.d_model)

        # prepend CLS: (B*T, 1, d_model)
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # → (B*T, N+1, d_model)

        # apply detection transformer
        x = self.transformer_det(x)            # → (B*T, N+1, d_model)

        # extract CLS output: (B*T, d_model)
        cls_det = x[:, 0, :]

        # reshape back to (B, T, d_model)
        cls_det = cls_det.view(B, T, self.d_model)

        # time‐level transformer over the T slots
        # input: (B, T, d_model) → output: (B, T, d_model)
        ctx_cls = self.transformer_time(cls_det)

        # classify each time‐step: (B, T, d_model) → (B, T, 7)
        logits = self.classifier(ctx_cls)

        return logits