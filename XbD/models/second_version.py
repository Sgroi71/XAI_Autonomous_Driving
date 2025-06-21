import torch
import torch.nn as nn
import torch.nn.functional as F

class XbD_SecondVersion_Hook(nn.Module):
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

        # Register hooks after transformer is built
        self._register_attention_hooks()

    def _init_params(self):
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _register_attention_hooks(self):
        """
        Register hooks on all MultiheadAttention layers to extract attention weights
        """
        for idx, layer in enumerate(self.transformer.layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    if self.return_attn:
                        attn_output, attn_weights = output
                        # attn_weights: (B*T, num_heads, N+1, N+1)
                        cls_to_det = attn_weights[:, :, 0, 1:]  # (B*T, num_heads, N)
                        if self.average_heads:
                            cls_to_det = cls_to_det.mean(dim=1)  # (B*T, N)
                        self._attn_maps.append(cls_to_det)
                return hook

            layer.self_attn.register_forward_hook(make_hook(idx))

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

        # Forward through transformer
        x_enc = self.transformer(det_seq, src_key_padding_mask=key_pad_det)

        # Classification head
        cls_out = x_enc[:, 0, :]  # (B*T, d_model)
        cls_out = self.cls_drop(cls_out)
        logits = self.classifier(cls_out).view(B, T, -1)  # (B, T, 7)

        if return_attn:
            # Re-shape attention to (B, T, ...) from (B*T, ...)
            attn = [
                m.view(B, T, -1) if m.dim() == 2 else m.view(B, T, m.shape[1], -1)
                for m in self._attn_maps
            ]
            return logits, attn
        return logits
