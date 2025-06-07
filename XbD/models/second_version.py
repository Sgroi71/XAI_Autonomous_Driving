# Here version with transformer over the N detections with a final a CLS token
# Then, the final classification is done with a linear layer over the CLS token.

# This is a new idea for the XbD model, which uses a transformer to process the sequence of detections.

import torch
import torch.nn as nn

class XbD_SecondVersion(nn.Module):
    def __init__(self,
                 N: int,
                 num_classes: int = 41,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 2):
        """
        Args:
            d_model:        embedding dimension for each detection
            N:              number of detections per time step
            nhead:          number of attention heads
            num_layers:     number of transformer‐encoder layers
        """
        super().__init__()
        self.d_model = d_model
        self.N = N

        # project num_classes→d_model
        self.up_proj = nn.Linear(num_classes, d_model)

        # a single learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # transformer encoder (batch_first so input is (B*T, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final classifier from d_model → 7
        self.classifier = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes)
        Returns:
            logits: Tensor of shape (B, T, num_classes)
        """
        B, T, N, _ = x.shape
        assert N == self.N

        # up‐project each detection: (B*T*N, num_classes) → (B*T*N, d_model)
        x = x.view(B * T * N, -1)
        x = self.up_proj(x)
        x = torch.relu(x)
        # → (B, T, N, d_model)
        x = x.view(B, T, N, self.d_model)

        # flatten batch & time so we can run one big transformer:
        # (B*T, N, d_model)
        x = x.view(B * T, N, self.d_model)

        # prepend a [CLS] token at position 0 for each (B*T) sequence
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)              # (B*T, N+1, d_model)

        # transformer over length=(N+1)
        x = self.transformer(x)                            # (B*T, N+1, d_model)

        # take the [CLS] output at index 0
        cls_out = x[:, 0, :]                               # (B*T, d_model)

        # classify and reshape back to (B, T, num_classes)
        logits = self.classifier(cls_out)                  # (B*T, num_classes)
        return logits.view(B, T, -1)