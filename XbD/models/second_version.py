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
        layer_idx: int = -1,
        average_attn: bool = True
    ):
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes)
            return_attn: if True, also return attention scores from CLS to detections
            layer_idx: which transformer layer to extract attention from
            average_attn: if True, average over heads
        Returns:
            logits: Tensor of shape (B, T, 7)
            attn_scores (optional): Tensor of shape (B, T, N) or (B, T, num_heads, N)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got N={N}"
        device = x.device

        valid = (x.abs().sum(dim=-1) != 0)
        valid = valid.view(B * T, N)
        pad_mask = ~valid

        x_flat = x.view(B * T * N, C)
        x_flat = self.up_proj(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det = x_flat.view(B * T, N, self.d_model)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        seq = torch.cat([cls_tokens, x_det], dim=1)

        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_mask = torch.cat([cls_pad, pad_mask], dim=1)

        if return_attn:
            seq_per = seq.transpose(0, 1)
            layer = self.transformer.layers[layer_idx] # get the specified layer, default is the last layer
            attn_out, attn_weights = layer.self_attn(
                seq_per,
                seq_per,
                seq_per,
                key_padding_mask=key_pad_mask,
                need_weights=True,
                average_attn_weights=False
            )
            num_heads = layer.self_attn.num_heads
            batch_heads, tgt_len, src_len = attn_weights.size()
            batch = batch_heads // num_heads
            attn_weights = attn_weights.view(batch, num_heads, tgt_len, src_len)
            cls_to_det = attn_weights[:, :, 0, 1:] # attention from CLS to detections
            cls_to_det = cls_to_det.view(B, T, num_heads, N)
            if average_attn:
                cls_to_det = cls_to_det.mean(dim=2) # average over heads

        x_enc = self.transformer(seq, src_key_padding_mask=key_pad_mask)

        cls_out = x_enc[:, 0, :]
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out).view(B, T, -1)

        if return_attn:
            return logits, cls_to_det
        return logits