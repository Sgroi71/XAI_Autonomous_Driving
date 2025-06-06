import torch
import torch.nn as nn

class XbD_FirstVersion(nn.Module):
    def __init__(self, num_classes: int, N: int):
        """
        Args:
            d_model: dimensionality to project each 41‐dim vector into
            N: number of objects per time step (so that lin2’s input size = N*d_model)
        """
        super().__init__()
        self.N = N
        self.num_classes = num_classes

        # Second linear: (N * d_model) → 7
        self.lin2 = nn.Linear(N * self.num_classes, 7)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Tensor of shape (batch_size, T, N, num_classes)

        Returns:
            logits: Tensor of shape (batch_size, T, 7)
        """
        batch_size, T, N, _ = labels.shape
        assert N == self.N, f"Expected N={self.N}, but got N={N}"

        # Flatten (B, T, N, num_classes) → (B, T, N*num_classes)
        x = labels.reshape(batch_size, T, N * self.num_classes)

        # 5) Apply lin2 to each time step. First, merge (B, T, N*self.num_classes) → (B*T, N*self.num_classes)
        x = x.view(batch_size * T, N * self.num_classes)  # → (B*T, N*self.num_classes)
        logits = self.lin2(x)                            # → (B*T, 7)

        # 6) Reshape back to (B, T, 7)
        logits = logits.view(batch_size, T, 7)
        return logits