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
        self.return_attn = False
        self.average_heads = True
        self._attn_maps = []

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
            average_heads: if True, average over heads

        Returns:
            logits: Tensor of shape (B, T, 7)
            attn_scores (optional): List of Tensors (B, T, N) or (B, T, num_heads, N)
        """
        B, T, N, C = x.shape
        device = x.device

        self.return_attn = return_attn
        self.average_heads = average_heads
        self._attn_maps = []

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
        cls_tokens = self.cls_token.expand(B * T, 1, self.d_model)
        det_seq = torch.cat([cls_tokens, x_det], dim=1)

        # Key padding mask
        cls_pad = torch.zeros((B * T, 1), dtype=torch.bool, device=device)
        key_pad_det = torch.cat([cls_pad, det_pad], dim=1)

        # Manual version
        if return_attn:
            det_attn = []
            det_seq_per_layer = det_seq
            for layer_idx, det_layer in enumerate(self.transformer.layers):
                #print("Extracting attention maps for Detection Layer:", layer_idx)
                # --- Forward through self_attn and save attn map ---
                src2, attn_per_head = det_layer.self_attn(
                    det_seq_per_layer, det_seq_per_layer, det_seq_per_layer,
                    key_padding_mask=key_pad_det,
                    need_weights=True,
                    average_attn_weights=False
                )
                num_heads_det = det_layer.self_attn.num_heads
                # Extract attention from cls_token to each detection token (exclude cls_token itself)
                det_map = attn_per_head[:, :, 0, 1:]
                det_map = det_map.view(B, T, num_heads_det, N)
                #print("Layer", layer_idx, "det_map shape:", det_map.shape)
                if average_heads:
                    det_attn.append(det_map.mean(dim=2))
                else:
                    det_attn.append(det_map)
                # --- Continue through the rest of the layer ---
                det_seq_per_layer = det_seq_per_layer + det_layer.dropout1(src2)
                det_seq_per_layer = det_layer.norm1(det_seq_per_layer)
                src2 = det_layer.linear2(det_layer.dropout(det_layer.activation(det_layer.linear1(det_seq_per_layer))))
                det_seq_per_layer = det_seq_per_layer + det_layer.dropout2(src2)
                det_seq_per_layer = det_layer.norm2(det_seq_per_layer)


        # Forward through transformer
        out_det = self.transformer(det_seq, src_key_padding_mask=key_pad_det)

        if return_attn:
            # Assert that manual and automatic forwarding are the same, and if thats not the case
            #   panic. We need to be consistent in what we give in input in both cases
            assert torch.allclose(
                out_det,
                det_seq_per_layer,
                atol=1
            ), "Manual and automatic transformer outputs differ!"

        # Classification head
        cls_out = out_det[:, 0, :]  # (B*T, d_model)
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out).view(B, T, -1)  # (B, T, 7)

        if return_attn:
            return logits, det_attn
        return logits
