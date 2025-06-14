import torch
import torch.nn as nn
import torch.nn.functional as F

class XbD_SecondVersion(nn.Module):
    def __init__(
        self,
        N: int,
        num_classes: int = 41,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        self.N = N

        self.up_proj = nn.Linear(num_classes, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
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
        assert N == self.N, f"Expected N={self.N}, got N={N}"
        device = x.device

        valid = (x.abs().sum(dim=-1) != 0)
        valid = valid.view(B * T, N)
        pad_mask = ~valid

        x = x.view(B * T * N, C)
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.proj_norm(x)
        x = self.proj_drop(x)
        x = x.view(B * T, N, self.d_model)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_pad, pad_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        cls_out = x[:, 0, :]
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out)

        return logits.view(B, T, -1)