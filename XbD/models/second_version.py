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

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        average_attn: bool = True
    ):
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes)
            return_attn: if True, also return attention scores from CLS to detections
            average_attn: if True, average over heads and layers

        Returns:
            logits: Tensor of shape (B, T, 7)
            attn_scores (optional): Tensor of shape (B, T, N)
        """
        B, T, N, C = x.shape
        device = x.device

        # Prepare padding mask
        valid = (x.abs().sum(dim=-1) != 0)  # shape (B, T, N)
        pad_mask = ~valid.view(B * T, N)  # shape (B*T, N)

        # Projection
        x_flat = x.view(B * T * N, C)
        x_flat = self.up_proj(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det = x_flat.view(B * T, N, self.d_model)  # (B*T, N, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, d_model)
        seq = torch.cat([cls_tokens, x_det], dim=1)        # (B*T, N+1, d_model)

        # Extend pad mask
        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_mask = torch.cat([cls_pad, pad_mask], dim=1)  # (B*T, N+1)

        all_cls_attn = []

        if return_attn:
            src = seq
            for layer in self.transformer.layers:
                src_per = src  # (B*T, N+1, d_model)

                # Get self-attention weights
                attn_out, attn_weights = layer.self_attn(
                    src_per,
                    src_per,
                    src_per,
                    key_padding_mask=key_pad_mask,
                    need_weights=True,
                    average_attn_weights=False
                )  # attn_weights: (B*T * num_heads, N+1, N+1)

                num_heads = layer.self_attn.num_heads
                total_batch = attn_weights.size(0)
                assert total_batch % num_heads == 0
                b_t = total_batch // num_heads

                attn_weights = attn_weights.view(b_t, num_heads, N + 1, N + 1)  # (B*T, H, S, S)
                cls_attn = attn_weights[:, :, 0, 1:]  # from CLS to detections -> (B*T, H, N)
                all_cls_attn.append(cls_attn)

                src = layer(src_per, src_key_padding_mask=key_pad_mask)

            # Stack and average
            attn_tensor = torch.stack(all_cls_attn, dim=0)  # (L, B*T, H, N)
            if average_attn:
                attn_tensor = attn_tensor.mean(dim=(0, 2))  # (B*T, N)
            else:
                attn_tensor = attn_tensor.permute(1, 2, 0, 3)  # (B*T, H, L, N)

            # Reshape attention to (B, T, N)
            attn_tensor = attn_tensor.view(B, T, N)

        # Classification head
        x_enc = self.transformer(seq, src_key_padding_mask=key_pad_mask)
        cls_out = x_enc[:, 0, :]  # (B*T, d_model)
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out).view(B, T, -1)  # (B, T, 7)

        if return_attn:
            return logits, attn_tensor
        return logits

