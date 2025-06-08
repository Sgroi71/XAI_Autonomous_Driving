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
                 dropout: float = 0.1):
        super().__init__()
        self.N = N
        self.d_model = d_model

        # 1) project each detection → d_model
        self.up_proj    = nn.Linear(num_classes, d_model)
        self.proj_norm  = nn.LayerNorm(d_model)
        self.proj_drop  = nn.Dropout(dropout)

        # 2) learnable [CLS] token for detection transformer
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) detection‐level transformer (over N+1 tokens: CLS + detections)
        det_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_det,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_det = nn.TransformerEncoder(det_layer, num_layers=num_layers_det)

        # 4) time‐level transformer (over T CLS embeddings)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_time,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_time = nn.TransformerEncoder(time_layer, num_layers=num_layers_time)

        # 5) classifier on each time‐step’s CLS
        self.cls_drop   = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes), zero-padded where invalid
        Returns:
            logits: Tensor of shape (B, T, 7)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        device = x.device

        # —— build detection padding mask —— 
        # valid_det[b,t,n] = True if any feature non-zero
        valid_det = (x.abs().sum(dim=-1) != 0)      # (B, T, N)
        valid_det = valid_det.view(B*T, N)         # (B*T, N)
        det_pad = ~valid_det                       # True where padded

        # —— project & regularize detections ——
        x_flat = x.view(B*T*N, C)
        x_flat = self.up_proj(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det = x_flat.view(B*T, N, self.d_model)  # (B*T, N, d_model)

        # —— prepend CLS tokens ——
        cls_tokens = self.cls_token.expand(B*T, -1, -1)   # (B*T, 1, d_model)
        det_seq = torch.cat([cls_tokens, x_det], dim=1)   # (B*T, N+1, d_model)

        # —— combine into key_padding_mask for det transformer ——
        cls_pad = torch.zeros((B*T, 1), dtype=torch.bool, device=device)
        key_pad_det = torch.cat([cls_pad, det_pad], dim=1)  # (B*T, N+1)

        # —— detection‐level transformer with mask ——
        out_det = self.transformer_det(det_seq,
                                       src_key_padding_mask=key_pad_det)  # (B*T, N+1, d_model)

        # —— extract CLS per time‐step ——
        cls_det = out_det[:, 0, :]                          # (B*T, d_model)
        cls_det = cls_det.view(B, T, self.d_model)          # (B, T, d_model)

        # —— build time padding mask —— 
        # a time-step is valid if it has at least one valid detection
        valid_ts = valid_det.view(B, T, N).any(dim=-1)      # (B, T)
        time_pad = ~valid_ts                                # True where entire frame is padded

        # —— time‐level transformer with mask ——
        out_time = self.transformer_time(cls_det,
                                         src_key_padding_mask=time_pad)  # (B, T, d_model)

        # —— classify each time‐step’s CLS ——
        out_time = self.cls_drop(out_time)
        logits = self.classifier(out_time)                  # (B, T, 7)

        return logits