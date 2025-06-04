import torch
from torch.utils.data import Dataset

class FakeDataset(Dataset):
    def __init__(self, length: int, T: int = 1, N: int = 5):
        """
        Args:
            length: total number of samples in the dataset
            T: temporal dimension (fixed to 1 here)
            N: number of “objects” or entries per time step (fixed to 5 here)
        """
        super().__init__()
        self.length = length
        self.T = T
        self.N = N

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ego_labels: a single integer in [0, 6]
        ego_label = torch.randint(low=0, high=7, size=(1,)).item()

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
                # Generate a random permutation of [0..30], take first k, then shift by +10
                perm = torch.randperm(31)[:k] + 10
                labels[t, n, perm] = 1.0

        return {
            "ego_labels": ego_label,   # int 0..6
            "labels": labels          # Tensor of shape (T, N, 41), binary‐encoded
        }

# TODO EXAMPLE ... REMOVE THIS
if __name__ == "__main__":
    dataset = FakeDataset(length=100, T=1, N=5)
    sample = dataset[0]
    print("ego_labels:", sample["ego_labels"])           # e.g. 2
    print("labels shape:", sample["labels"].shape)       # torch.Size([1, 5, 41])
    print("labels[0,0]:", sample["labels"][0, 0])        # e.g. tensor([...], size=41)