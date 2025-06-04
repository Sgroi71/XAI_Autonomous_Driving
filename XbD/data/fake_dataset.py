import torch
from torch.utils.data import Dataset

class FakeDataset(Dataset):
    def __init__(self, length: int, T: int = 1, N: int = 5):
        """
        Args:
            length: total number of samples in the dataset
            T: temporal dimension
            N: number of “objects” or entries per time step
        """
        super().__init__()
        self.length = length
        self.T = T
        self.N = N

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ego_labels: one integer in [0, 6] per time step → shape (T,)
        ego_labels = torch.randint(low=0, high=7, size=(self.T,))

        # labels: shape (T, N, 41), where:
        #   • first 10 entries (0–9) are one-hot (exactly one “1”)
        #   • next 31 entries (10–40) are multi-hot with 3–5 active “1”s
        labels = torch.zeros(self.T, self.N, 41, dtype=torch.float)

        for t in range(self.T):
            for n in range(self.N):
                # 1) One‐hot among indices 0..9
                one_hot_index = torch.randint(low=0, high=10, size=(1,)).item()
                labels[t, n, one_hot_index] = 1.0

                # 2) Multi‐hot among indices 10..40: choose between 3 and 5 distinct positions
                k = torch.randint(low=3, high=6, size=(1,)).item()  # random int in {3,4,5}
                perm = torch.randperm(31)[:k] + 10
                labels[t, n, perm] = 1.0

        return {
            "ego_labels": ego_labels,   # Tensor of shape (T,)
            "labels": labels            # Tensor of shape (T, N, 41), binary‐encoded
        }

# TODO EXAMPLE ... REMOVE THIS
if __name__ == "__main__":
    dataset = FakeDataset(length=100, T=8, N=5)
    sample = dataset[0]
    print("ego_labels:", sample["ego_labels"], "→ shape:", sample["ego_labels"].shape)
    # e.g. tensor([3, 1])  → shape: torch.Size([2])
    print("labels shape:", sample["labels"].shape)
    # torch.Size([2, 5, 41])
    print("labels[0,0]:", sample["labels"][0, 0])
    # e.g. tensor([...], size=41)