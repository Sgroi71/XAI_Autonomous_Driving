import torch
import torch.nn as nn

class XbD_FirstVersion(nn.Module):
    def __init__(self, d_model: int, N: int):
        """
        Args:
            d_model: dimensionality to project each 41‐dim vector into
            N: number of objects per time step (so that lin2’s input size = N*d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.N = N

        # First linear: 41 → d_model
        self.lin1 = nn.Linear(41, d_model)
        # Second linear: (N * d_model) → 7
        self.lin2 = nn.Linear(N * d_model, 7)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Tensor of shape (batch_size, T, N, 41)

        Returns:
            logits: Tensor of shape (batch_size, T, 7)
        """
        batch_size, T, N, _ = labels.shape
        assert N == self.N, f"Expected N={self.N}, but got N={N}"

        # 1) Flatten so that we can apply lin1 to each (41‐dim) vector:
        #    (B, T, N, 41) → (B * T * N, 41)
        x = labels.view(batch_size * T * N, 41)

        # 2) First projection: (B*T*N, 41) → (B*T*N, d_model)
        x = self.lin1(x)
        x = torch.relu(x)

        # 3) Reshape back to (B, T, N, d_model)
        x = x.view(batch_size, T, N, self.d_model)

        # 4) Flatten the N and d_model dims:
        #    (B, T, N, d_model) → (B, T, N*d_model)
        x = x.reshape(batch_size, T, N * self.d_model)

        # 5) Apply lin2 to each time step. First, merge (B, T, N*d_model) → (B*T, N*d_model)
        x = x.view(batch_size * T, N * self.d_model)  # → (B*T, N*d_model)
        logits = self.lin2(x)                            # → (B*T, 7)

        # 6) Reshape back to (B, T, 7)
        logits = logits.view(batch_size, T, 7)
        return logits