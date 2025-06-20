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

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        det_layer_idx: int = -1,
        time_layer_idx: int = -1,
        average_heads: bool = True
    ):
        """
        Args:
            x: Tensor of shape (B, T, N, num_classes)
            return_attn: if True, return both detection and time attention maps
            det_layer_idx: layer index for detection transformer
            time_layer_idx: layer index for time transformer
            average_heads: if True, average attention over heads
        Returns:
            logits: (B, T, 7)
            det_attn (optional): (B, T, N) or (B, T, num_heads_det, N)
            time_attn (optional): (B, T, T) or (B, num_heads_time, T, T)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        device = x.device

        valid_det = (x.abs().sum(dim=-1) != 0).view(B * T, N)
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

        det_attn = []

        # same as version 2
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

        out_det = self.transformer_det(det_seq, src_key_padding_mask=key_pad_det)
        cls_det = out_det[:, 0, :].view(B, T, self.d_model)

        time_attn = []

        if return_attn:
            time_attn = []
            time_seq_per_layer = cls_det
            for layer_idx, time_layer in enumerate(self.transformer_time.layers):
                print("Extracting attention maps for Time Layer:", layer_idx)
                time_seq_per_layer, attn_per_head = time_layer.self_attn(
                    time_seq_per_layer, time_seq_per_layer, time_seq_per_layer,
                    need_weights=True,
                    average_attn_weights=False
                )
                num_heads_time = time_layer.self_attn.num_heads
                # attn_per_head: (B, num_heads_time, T, T)
                attn_map = attn_per_head.view(B, num_heads_time, T, T)
                if average_heads:
                    time_attn.append(attn_map.mean(dim=1))
                else:
                    time_attn.append(attn_map)

        out_time = self.transformer_time(cls_det)
        out_time = self.cls_drop(out_time)
        logits = self.classifier(out_time)

        if return_attn:
            return logits, det_attn, time_attn
        return logits