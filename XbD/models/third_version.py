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

    def _generate_causal_mask(self, T: int, device) -> torch.Tensor:          # <-- added for causal attention
        """
        Returns an upper-triangular boolean mask for causal attention.
        Shape: (T, T) with True where *attention should be blocked*.
        """
        return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    # ------------------------------------------------------------------

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
            det_attn (optional): (B, T, N)            or (B, T, num_heads_det, N)
            time_attn(optional): (B, T, T)            or (B, num_heads_time, T, T)
        """
        B, T, N, C = x.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        device = x.device

        # ================ Detection branch (unchanged) =================
        valid_det = (x.abs().sum(dim=-1) != 0).view(B * T, N)
        det_pad   = ~valid_det

        x_flat = self.up_proj(x.view(B * T * N, C))
        x_flat = F.relu(x_flat)
        x_flat = self.proj_norm(x_flat)
        x_flat = self.proj_drop(x_flat)
        x_det  = x_flat.view(B * T, N, self.d_model)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        det_seq    = torch.cat([cls_tokens, x_det], dim=1)

        cls_pad     = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_det = torch.cat([cls_pad, det_pad], dim=1)

        det_attn = []
        if return_attn:
            det_seq_per_layer = det_seq
            for layer_idx, det_layer in enumerate(self.transformer_det.layers):
                src2, attn_per_head = det_layer.self_attn(
                    det_seq_per_layer, det_seq_per_layer, det_seq_per_layer,
                    key_padding_mask=key_pad_det,
                    need_weights=True,
                    average_attn_weights=False
                )
                attn_sum = attn_per_head.sum(dim=-1)
                assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4)
                num_heads_det = det_layer.self_attn.num_heads
                det_map = attn_per_head[:, :, 0, 1:].view(B, T, num_heads_det, N)
                det_attn.append(det_map.mean(dim=2) if average_heads else det_map)

                det_seq_per_layer = det_seq_per_layer + det_layer.dropout1(src2)
                det_seq_per_layer = det_layer.norm1(det_seq_per_layer)
                src2 = det_layer.linear2(det_layer.dropout(
                    det_layer.activation(det_layer.linear1(det_seq_per_layer))))
                det_seq_per_layer = det_seq_per_layer + det_layer.dropout2(src2)
                det_seq_per_layer = det_layer.norm2(det_seq_per_layer)

        out_det = self.transformer_det(det_seq, src_key_padding_mask=key_pad_det)
        assert torch.allclose(out_det, det_seq_per_layer, atol=1e-6)
        cls_det = out_det[:, 0, :].view(B, T, self.d_model)

        # ================= Temporal branch (CAUSAL) ================
        time_mask = self._generate_causal_mask(T, device)                     # <-- added

        time_attn = []
        if return_attn:
            time_seq_per_layer = cls_det
            for layer_idx, time_layer in enumerate(self.transformer_time.layers):
                src2, attn_per_head = time_layer.self_attn(
                    time_seq_per_layer, time_seq_per_layer, time_seq_per_layer,
                    attn_mask=time_mask,                                         # <-- changed
                    need_weights=True,
                    average_attn_weights=False
                )
                attn_sum = attn_per_head.sum(dim=-1)
                assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4)
                num_heads_time = time_layer.self_attn.num_heads
                attn_map = attn_per_head.view(B, num_heads_time, T, T)
                time_attn.append(attn_map.mean(dim=1) if average_heads else attn_map)

                time_seq_per_layer = time_seq_per_layer + time_layer.dropout1(src2)
                time_seq_per_layer = time_layer.norm1(time_seq_per_layer)
                src2 = time_layer.linear2(time_layer.dropout(
                    time_layer.activation(time_layer.linear1(time_seq_per_layer))))
                time_seq_per_layer = time_seq_per_layer + time_layer.dropout2(src2)
                time_seq_per_layer = time_layer.norm2(time_seq_per_layer)

        # automatic path (must use the same mask!)
        out_time = self.transformer_time(cls_det, mask=time_mask)             # <-- changed
        if return_attn:
            assert torch.allclose(out_time, time_seq_per_layer, atol=1e-6)

        out_time = self.cls_drop(out_time)
        logits   = self.classifier(out_time)

        if return_attn:
            return logits, det_attn, time_attn
        return logits