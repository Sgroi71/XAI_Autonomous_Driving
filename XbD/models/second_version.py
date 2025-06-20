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
        average_heads: bool = True
    ):
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes)
            return_attn: if True, also return attention scores from CLS to detections
            average_heads: if True, average over heads and layers

        Returns:
            logits: Tensor of shape (B, T, 7)
            attn_scores (optional): Tensor of shape (B, T, N)
        """
        B, T, N, C = x.shape
        device = x.device

        # Prepare padding mask
        valid_det = (x.abs().sum(dim=-1) != 0).view(B * T, N)
        det_pad = ~valid_det

        # Projection
        x_flat = x.view(B * T * N, C)
        x_flat = self.up_proj(x_flat)
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det = x_flat.view(B * T, N, self.d_model)  # (B*T, N, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, d_model)
        det_seq = torch.cat([cls_tokens, x_det], dim=1)        # (B*T, N+1, d_model)

        # Extend pad mask
        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_det = torch.cat([cls_pad, det_pad], dim=1)

        if return_attn:
            det_attn = []
            det_seq_per_layer = det_seq
            for layer_idx, det_layer in enumerate(self.transformer_det.layers):
                print("Extracting attention maps for Detection Layer:", layer_idx)
                det_seq_per_layer, attn_per_head = det_layer.self_attn(
                    det_seq_per_layer, det_seq_per_layer, det_seq_per_layer,
                    key_padding_mask=key_pad_det,
                    need_weights=True,
                    average_attn_weights=False
                )
                print("Layer", layer_idx, "attention shape:", attn_per_head.shape)
                num_heads_det = det_layer.self_attn.num_heads
                # Extract attention from cls_token to each detection token (exclude cls_token itself)
                det_map = attn_per_head[:, :, 0, 1:]
                det_map = det_map.view(B, T, num_heads_det, N)
                if average_heads:
                    det_attn.append(det_map.mean(dim=2))
                else:
                    det_attn.append(det_map)

        

        # Classification head
        x_enc = self.transformer(det_seq, src_key_padding_mask=key_pad_det)
        cls_out = x_enc[:, 0, :]  # (B*T, d_model)
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out).view(B, T, -1)  # (B, T, 7)

        if return_attn:
            return logits, det_attn
        return logits

