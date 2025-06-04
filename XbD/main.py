import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # for progress bar

from data.fake_dataset import FakeDataset
from models.first_version import XbD_FirstVersion
from utils_loss import ego_loss


def get_device(use_mps: bool = False):
    """
    Return the appropriate device. If use_mps is True, attempt to use Apple’s MPS backend.
    Otherwise, use CUDA if available, else CPU.
    """
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one full epoch.
    Returns the average loss over all batches.
    """
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        labels_tensor = batch["labels"].to(device)       # → (B, T, N, 41)
        ego_targets = batch["ego_labels"].to(device)     # → (B, T)

        # Forward pass: logits shape = (B, T, 7)
        logits = model(labels_tensor)

        # Flatten both time and batch dims to compute CrossEntropyLoss:
        B, T, _ = logits.shape
        logits_flat = logits.view(B * T, 7)              # → (B*T, 7)
        targets_flat = ego_targets.view(B * T)           # → (B*T,)

        loss = criterion(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def train(model, dataloader, criterion, optimizer, device, num_epochs):
    """
    Train the model for a given number of epochs, printing the average loss per epoch.
    """
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {avg_loss:.4f}")


def evaluate(model, dataloader, device):
    print()
    # TODO: implement evaluation to compute mAP per class and total mAP.


def inference_on_sample(model, batch, device, sample_idx: int = 0):
    """
    Given a single batch dict (with keys "labels" and optionally "ego_labels"),
    pick one sample (index = sample_idx) from the batch and return:
      - logits_sample: Tensor of shape (T, 7)
      - preds_sample: Tensor of shape (T,)
      - targets_sample: Tensor of shape (T,) if "ego_labels" present
    """
    model.eval()
    # Extract the single sample at index sample_idx
    labels_tensor = batch["labels"][sample_idx].unsqueeze(0).to(device)  # → (1, T, N, 41)
    logits = model(labels_tensor)                                        # → (1, T, 7)
    preds = torch.argmax(logits, dim=2)                                  # → (1, T)

    # Remove the batch dimension
    logits_sample = logits.squeeze(0).cpu()    # → (T, 7)
    preds_sample = preds.squeeze(0).cpu()      # → (T,)

    if "ego_labels" in batch:
        targets = batch["ego_labels"][sample_idx]  # → (T,)
        return logits_sample, preds_sample, targets
    return logits_sample, preds_sample


def main():
    # ----------------------------
    # Hyperparameters & Settings
    # ----------------------------
    length = 1_000_000       # number of samples in FakeDataset
    T = 1                    # temporal dimension
    N = 5                    # number of objects per time step
    d_model = 64             # projection dimension
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    use_mps = False          # set to True to use Mac GPU

    # ----------------------------
    # Device Configuration
    # ----------------------------
    device = get_device(use_mps=use_mps)
    print(f"Using device: {device}")

    # ----------------------------
    # Dataset & DataLoader
    # ----------------------------
    dataset = FakeDataset(length=length, T=T, N=N)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    # ----------------------------
    # Model, Loss, Optimizer
    # ----------------------------
    model = XbD_FirstVersion(d_model=d_model, N=N).to(device)
    criterion = ego_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------------
    # Training
    # ----------------------------
    train(model, dataloader, criterion, optimizer, device, num_epochs)

    # ----------------------------
    # Example Inference on a Sample
    # ----------------------------
    sample_batch = next(iter(dataloader))
    logits_s, preds_s, targets_s = inference_on_sample(model, sample_batch, device, sample_idx=0)

    print("\nExample Inference on One Sample (from first batch):")
    print("Predicted ego_labels (T):")
    print(preds_s.tolist())
    print("Ground-truth ego_labels (T):")
    print(targets_s.tolist())


if __name__ == "__main__":
    main()