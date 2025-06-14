import torch
import torch.nn as nn
import torch.nn.functional as F

class XbD_ThirdVersion(nn.Module):
    def __init__(
        self,
        N: int,
        num_classes: int = 41,
        d_model: int = 64,
        nhead_det: int = 2,
        num_layers_det: int = 1,
        nhead_time: int = 2,
        num_layers_time: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.up_proj = nn.Linear(num_classes, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

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

        self.cls_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 7)

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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

        valid_det = (x.abs().sum(dim=-1) != 0)
        valid_det = valid_det.view(B * T, N)
        det_pad = ~valid_det

        x_flat = x.view(B * T * N, C)
        x_flat = self.up_proj(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det = x_flat.view(B * T, N, self.d_model)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        det_seq = torch.cat([cls_tokens, x_det], dim=1)

        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_det = torch.cat([cls_pad, det_pad], dim=1)

        out_det = self.transformer_det(
            det_seq,
            src_key_padding_mask=key_pad_det
        )

        cls_det = out_det[:, 0, :].view(B, T, self.d_model)

        out_time = self.transformer_time(cls_det)

        out_time = self.cls_drop(out_time)
        logits = self.classifier(out_time)

        return logits