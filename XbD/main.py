# file: train.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # for progress bar

from data.fake_dataset import FakeDataset
from models.first_version import XbD_FirstVersion

def main():
    # Hyperparameters
    length = 1000000        # number of samples in FakeDataset
    T = 1                   # temporal dimension
    N = 5                   # number of objects per time step
    d_model = 64            # projection dimension
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # Device
    device = torch.device("mps")  # uncomment this line to use Mac GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    dataset = FakeDataset(length=length, T=T, N=N)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model, Loss, Optimizer
    model = XbD_FirstVersion(d_model=d_model, N=N).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in loop:
            labels_tensor = batch["labels"].to(device)      # → (B, T, N, 41)
            ego_targets = batch["ego_labels"].to(device)    # → (B,)

            # Forward pass: logits shape = (B, T, 7)
            logits = model(labels_tensor)

            # Since T=1, squeeze time dimension → (B, 7)
            logits_squeezed = logits.squeeze(1)

            # Compute loss
            loss = criterion(logits_squeezed, ego_targets)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {avg_loss:.4f}")

    # Example inference on a small batch
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_labels = sample_batch["labels"].to(device)  # (B, T, N, 41)
        logits = model(sample_labels)                      # (B, T, 7)
        preds = torch.argmax(logits, dim=2).squeeze(1)     # (B,)
        print("Predicted ego_labels for first batch:", preds.cpu().tolist())

if __name__ == "__main__":
    main()