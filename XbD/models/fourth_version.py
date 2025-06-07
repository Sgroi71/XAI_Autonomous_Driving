import torch
import torch.nn as nn


class FrameMemoryTransformer(nn.Module):
    """
    Inputs (per forward call):
        x_logits:   FloatTensor of shape (B, T, N, num_classes)
                    -- these are the per-box logits of length num_classes=41.
        prev_memory: Optional FloatTensor of shape (B, M, d_model), where
                     M = memory_size. If None, we assume “empty” past memory.

    Outputs:
        frame_logits:   FloatTensor of shape (B, T, num_ego_actions)
                        (num_ego_actions is 7 in your description)
        new_memory:     FloatTensor of shape (B, M, d_model) = the updated memory
                        (keeps at most `memory_size` of the concatenated [old || current]).
    """
    def __init__(
        self,
        num_classes: int = 41,
        d_model: int = 256,
        num_heads_spatial: int = 8,
        num_layers_spatial: int = 2,
        num_heads_temporal: int = 8,
        num_layers_temporal: int = 2,
        memory_size: int = 50,
        num_ego_actions: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.memory_size = memory_size
        self.num_ego_actions = num_ego_actions

        # 1) Project each detection’s 41-dim logits → d_model
        self.input_proj = nn.Linear(num_classes, d_model)

        # 2) Learnable spatial [CLS] token (one vector that gets prepended for each frame)
        #    We will expand it to (B*T, 1, d_model) at runtime.
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, d_model))

        # 3) Spatial Transformer Encoder (over N detections + 1 [CLS])
        encoder_layer_spatial = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads_spatial,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer_spatial,
            num_layers=num_layers_spatial
        )

        # 4) Temporal Transformer Encoder (over M + T frame-CLS tokens)
        encoder_layer_temporal = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads_temporal,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer_temporal,
            num_layers=num_layers_temporal
        )

        # 5) A small linear head to go from d_model → num_ego_actions (7)
        self.classifier = nn.Linear(d_model, num_ego_actions)


    def forward(
        self,
        x_logits: torch.Tensor,
        prev_memory: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor): # type: ignore
        """
        Args:
            x_logits:    (B, T, N, num_classes)
            prev_memory: (B, M, d_model) or None.

        Returns:
            frame_logits: (B, T, num_ego_actions)
            new_memory:   (B, M, d_model)
        """
        B, T, N, C = x_logits.shape
        assert C == self.num_classes, f"Expected num_classes={self.num_classes}, got {C}"

        # === 1) SPATIAL PART: run a transformer over each frame’s detections ===
        # 1.a) reshape so we can process all frames at once:
        #     from (B, T, N, C) → (B*T, N, C)
        x_flat = x_logits.view(B * T, N, C)

        # 1.b) project each detection: (B*T, N, C) → (B*T, N, d_model)
        x_proj = self.input_proj(x_flat)  # → (B*T, N, d_model)

        # 1.c) prepend a learnable [CLS]_spatial token to each (B*T) sequence
        #     cls_token: (1, 1, d_model) → expand to (B*T, 1, d_model)
        cls_tok = self.cls_token_spatial.expand(B * T, -1, -1)  # (B*T, 1, d_model)
        x_seq = torch.cat([cls_tok, x_proj], dim=1)  # (B*T, 1 + N, d_model)

        # 1.d) run the spatial transformer over (N+1) tokens
        #     output:   (B*T, N+1, d_model)
        spatial_out = self.spatial_transformer(x_seq)

        # 1.e) extract the [CLS]_spatial output (index 0 in dimension 1)
        #     → (B*T, d_model), then reshape → (B, T, d_model)
        cls_feat = spatial_out[:, 0, :].view(B, T, self.d_model)  # (B, T, d_model)

        # === 2) TEMPORAL PART: combine with memory, run another transformer, classify ===
        # 2.a) If no prev_memory is given, create an “empty” memory of shape (B, 0, d_model).
        if prev_memory is None:
            prev_memory = cls_feat.new_zeros((B, 0, self.d_model))

        # sanity-check shapes
        assert prev_memory.ndim == 3 and prev_memory.shape[0] == B
        M = prev_memory.shape[1]  # how many timesteps were in memory

        # 2.b) concatenate along sequence-length dimension: [B, M, d_model] + [B, T, d_model]
        #     → temporalseq shape = (B, M + T, d_model)
        temp_seq = torch.cat([prev_memory, cls_feat], dim=1)  # (B, M+T, d_model)

        # 2.c) run the temporal transformer over these (M+T) tokens
        #     → (B, M+T, d_model)
        temp_out = self.temporal_transformer(temp_seq)

        # 2.d) extract the *last T* outputs (corresponding to the current frames)
        #     → (B, T, d_model)
        #     (the first M positions are “old” memory; last T are “current”)
        out_current = temp_out[:, M : M + T, :]  # (B, T, d_model)

        # 2.e) classification head: (B, T, d_model) → (B, T, num_ego_actions)
        frame_logits = self.classifier(out_current)  # (B, T, 7)

        # === 3) UPDATE MEMORY: keep at most self.memory_size of [old || current CLS] tokens ===
        # We want new_memory = last M' tokens of (prev_memory || cls_feat), where M' = memory_size.
        cat_memory = torch.cat([prev_memory, cls_feat], dim=1)  # (B, M + T, d_model)
        if (M + T) <= self.memory_size:
            # we can keep them all
            new_memory = cat_memory  # (B, M+T, d_model)
        else:
            # drop the oldest, keep only the final `memory_size` positions
            new_memory = cat_memory[:, -self.memory_size :, :].contiguous()  # (B, memory_size, d_model)

        return frame_logits, new_memory
