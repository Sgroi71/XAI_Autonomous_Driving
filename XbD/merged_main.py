import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Model Imports ---
# Import all model versions
from models.first_version import XbD_FirstVersion
from models.second_version import XbD_SecondVersion
from models.third_version import XbD_ThirdVersion
from models.fourth_version import FrameMemoryTransformer

# --- Data and Loss Imports ---
from data.dataset_prediction import VideoDataset
from utils_loss import ego_loss

# --- Global Configuration ---
# Set the root directories for the project and data
ROOT = '/home/jovyan/python/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/jovyan/nfs/lsgroi/'

# ##############################################################################
# # --- SELECT MODEL VERSION HERE ---
# # Change this variable to 1, 2, 3, or 4 to switch between models
# ##############################################################################
MODEL_VERSION = 3
# ##############################################################################


# ##############################################################################
# # --- UTILITY AND EVALUATION FUNCTIONS ---
# # These functions are common across all model versions.
# ##############################################################################

def get_device():
    """Returns the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_ego(gts, dets, classes):
    """
    Evaluates ego-motion prediction performance using mean Average Precision (mAP).
    """
    ap_strs = []
    num_frames = gts.shape[0]
    print(f'Evaluating for {num_frames} frames')

    if num_frames < 1:
        return 0, [0, 0], ['no gts present', 'no gts present']

    ap_all = []
    sap = 0.0
    for cls_ind, class_name in enumerate(classes):
        scores = dets[:, cls_ind]
        istp = np.zeros_like(gts)
        istp[gts == cls_ind] = 1
        det_count = num_frames
        num_positives = np.sum(istp)
        cls_ap = get_class_ap_from_scores(scores, istp, num_positives)
        ap_all.append(cls_ap)
        sap += cls_ap
        ap_str = f"{class_name} : {num_positives} : {det_count} : {cls_ap}"
        ap_strs.append(ap_str)

    mAP = sap / len(classes)
    ap_strs.append(f'FRAME Mean AP:: {mAP:0.2f}')
    return mAP, ap_all, ap_strs

def get_class_ap_from_scores(scores, istp, num_positives):
    """Calculates the Average Precision for a single class."""
    if num_positives < 1:
        num_positives = 1
    argsort_scores = np.argsort(-scores)
    istp = istp[argsort_scores]
    fp = np.cumsum(istp == 0)
    tp = np.cumsum(istp == 1)
    fp = fp.astype(np.float64)
    tp = tp.astype(np.float64)
    recall = tp / float(num_positives)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    cls_ap = voc_ap(recall, precision)
    return cls_ap

def voc_ap(rec, prec, use_07_metric=False):
    """
    Computes the VOC Average Precision.
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap * 100

def save_model_weights(model: nn.Module, path: str):
    """Saves the model's state dictionary to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_model_weights(model_class, checkpoint_path: str, device, **model_kwargs):
    """Loads model weights from a checkpoint file."""
    model = model_class(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {checkpoint_path}")
    return model

# ##############################################################################
# # --- TRAINING & EVALUATION LOGIC (MODELS 1, 2, 3) ---
# ##############################################################################

def train_one_epoch_stateless(model, dataloader, criterion, optimizer, device):
    """
    Trains stateless models (versions 1, 2, 3) for one epoch.
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training (Stateless)", leave=False)

    for batch in loop:
        labels_tensor = batch["labels"].to(device)
        ego_targets = batch["ego_labels"].to(device)

        logits = model(labels_tensor)
        B, T, _ = logits.shape
        logits_flat = logits.view(B * T, 7)
        targets_flat = ego_targets.view(B * T)

        loss = criterion(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)

def evaluate_stateless(model, dataloader, device, criterion=None):
    """
    Evaluates stateless models (versions 1, 2, 3).
    """
    model.eval()
    all_gts, all_dets = [], []
    total_loss, num_batches = 0.0, 0
    classes = [str(i) for i in range(7)]

    with torch.no_grad():
        for batch in dataloader:
            labels_tensor = batch["labels"].to(device)
            ego_targets = batch["ego_labels"].to(device)
            
            logits = model(labels_tensor)
            preds = torch.sigmoid(logits)

            B, T, _ = logits.shape
            all_gts.append(ego_targets.view(-1).cpu().numpy())
            all_dets.append(preds.view(-1, 7).cpu().numpy())

            if criterion:
                loss = criterion(logits.view(B * T, 7), ego_targets.view(B * T))
                total_loss += loss.item()
                num_batches += 1

    if not all_gts:
        return None, None, ["No data for evaluation."]

    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)
    
    mAP, ap_all, ap_strs = evaluate_ego(gts, dets, classes)
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    return mAP, avg_loss, ap_strs

# ##############################################################################
# # --- TRAINING & EVALUATION LOGIC (MODEL 4 - MEMORY TRANSFORMER) ---
# ##############################################################################

def train_one_epoch_memory(model, dataloader, criterion, optimizer, device, actual_seq_len):
    """
    Trains the memory-based model (version 4) for one epoch.
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training (Memory)", leave=False)

    for batch in loop:
        B, T, _, _ = batch["labels"].shape
        num_slices = T // actual_seq_len
        
        prev_memory = None
        total_loss_for_clip = 0.0
        optimizer.zero_grad()

        for i in range(num_slices):
            start_idx, end_idx = i * actual_seq_len, (i + 1) * actual_seq_len
            labels_slice = batch["labels"][:, start_idx:end_idx].to(device)
            targets_slice = batch["ego_labels"][:, start_idx:end_idx].to(device)

            if prev_memory is not None:
                prev_memory = prev_memory.detach().to(device)

            logits, prev_memory = model(labels_slice, prev_memory)

            B_slice, T_slice, _ = logits.shape
            logits_flat = logits.view(B_slice * T_slice, 7)
            targets_flat = targets_slice.view(B_slice * T_slice)
            
            loss = criterion(logits_flat, targets_flat)
            total_loss_for_clip += loss

        total_loss_for_clip.backward()
        optimizer.step()
        
        running_loss += total_loss_for_clip.item()
        loop.set_postfix(loss=total_loss_for_clip.item())

    return running_loss / len(dataloader)


def evaluate_memory(model, dataloader, device, criterion=None, actual_seq_len=8):
    """
    Evaluates the memory-based model (version 4).
    """
    model.eval()
    all_gts, all_dets = [], []
    total_loss, num_clips = 0.0, 0
    classes = [str(i) for i in range(7)]

    with torch.no_grad():
        for batch in dataloader:
            B, T, _, _ = batch["labels"].shape
            num_slices = T // actual_seq_len
            prev_memory = None

            for i in range(num_slices):
                start_idx, end_idx = i * actual_seq_len, (i + 1) * actual_seq_len
                labels_slice = batch["labels"][:, start_idx:end_idx].to(device)
                targets_slice = batch["ego_labels"][:, start_idx:end_idx].to(device)

                if prev_memory is not None:
                    prev_memory = prev_memory.to(device)

                logits, prev_memory = model(labels_slice, prev_memory)
                preds = torch.sigmoid(logits)

                B_slice, T_slice, _ = logits.shape
                all_gts.append(targets_slice.view(-1).cpu().numpy())
                all_dets.append(preds.view(-1, 7).cpu().numpy())

                if criterion:
                    loss = criterion(logits.view(B_slice*T_slice, 7), targets_slice.view(B_slice*T_slice))
                    total_loss += loss.item()
            num_clips += 1

    if not all_gts:
        return None, None, ["No data for evaluation."]

    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)
    
    mAP, ap_all, ap_strs = evaluate_ego(gts, dets, classes)
    avg_loss = total_loss / (num_clips * num_slices) if num_clips > 0 else None
    return mAP, avg_loss, ap_strs


# ##############################################################################
# # --- UNIFIED TRAINING ORCHESTRATOR ---
# ##############################################################################

def train(model, model_version, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs, patience, actual_seq_len=None):
    """
    Main training loop that orchestrates training and validation based on model version.
    """
    best_mAP = -float('inf')
    epochs_no_improve = 0
    output_dir = f"{ROOT}XbD/results/version{model_version}"
    best_model_path = os.path.join(output_dir, "best_model_weights.pth")
    
    train_losses, val_losses, mAPs= [], [], []

    for epoch in range(1, num_epochs + 1):
        # --- Select Training Function ---
        if model_version == 4:
            avg_train_loss = train_one_epoch_memory(model, dataloader_train, criterion, optimizer, device, actual_seq_len)
        else:
            avg_train_loss = train_one_epoch_stateless(model, dataloader_train, criterion, optimizer, device)
        train_losses.append(avg_train_loss)
        print(f"\nEpoch [{epoch}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}")

        # --- Select Evaluation Function ---
        print("Evaluating on validation set...")
        if model_version == 4:
            mAP, avg_val_loss, ap_strs = evaluate_memory(model, dataloader_val, device, criterion, actual_seq_len)
        else:
            mAP, avg_val_loss, ap_strs = evaluate_stateless(model, dataloader_val, device, criterion)
        
        mAPs.append(mAP)
        val_losses.append(avg_val_loss)
        print(f"Validation mAP: {mAP:.4f}" if mAP is not None else "Validation mAP: None")
        print(f"Validation Loss: {avg_val_loss:.4f}" if avg_val_loss is not None else "Validation Loss: None")

        # --- Check for Improvement and Save Model ---
        if mAP is not None and mAP > best_mAP:
            best_mAP = mAP
            epochs_no_improve = 0
            save_model_weights(model, best_model_path)
            ap_file_path = os.path.join(output_dir, "best_ap_strs.txt")
            with open(ap_file_path, "w") as f:
                f.write("\n".join(ap_strs))
            print(f"New best mAP: {best_mAP:.4f} -> Model and AP strings saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in mAP for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # --- Plotting Results ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training & Validation Loss (Version {model_version})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(mAPs, label='Validation mAP')
    plt.title(f'Validation mAP Over Epochs (Version {model_version})')
    plt.xlabel('Epoch'); plt.ylabel('mAP'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mAP_plot.png"))
    print(f"Plots saved to {output_dir}")

# ##############################################################################
# # --- MAIN EXECUTION BLOCK ---
# ##############################################################################

def main():
    """
    Main function to set up and run the experiment.
    """
    print(f"--- Running Experiment for Model Version: {MODEL_VERSION} ---")
    
    # ----------------------------
    # Hyperparameters & Settings
    # ----------------------------
    # Default values
    T = 8
    N = 10
    batch_size = 1024
    num_epochs = 500
    learning_rate = 1e-4
    patience = 100
    actual_input_len = None # Specific to model 4

    kwargs = {
        "d_model": 64,
        "nhead_det": 2,
        "num_layers_det": 1,
        "nhead_time": 2,
        "num_layers_time": 1,
        "dropout": 0.1,
    }

    # Version-specific settings
    if MODEL_VERSION == 4:
        T = 48
        N = 10
        actual_input_len = 8
        batch_size = 256
        
    # ----------------------------
    # Device Configuration
    # ----------------------------
    device = get_device()
    print(f"Using device: {device}")

    # ----------------------------
    # Dataset & DataLoader
    # ----------------------------
    class Args:
        ANCHOR_TYPE = 'default'
        DATASET = 'road'
        SEQ_LEN = T
        SUBSETS = ['train_3']
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
        PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadal-h3x3x3/detections-30-08-50')
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41
        if MODEL_VERSION == 4:
            ACTUAL_SEQ_LEN = actual_input_len

    args = Args()
    
    dataset_train = VideoDataset(args, train=True, skip_step=args.SEQ_LEN)
    args.SUBSETS=['val_3']
    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)

    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=device.type == "cuda")
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=device.type == "cuda")
    
    # ----------------------------
    # Model, Loss, Optimizer
    # ----------------------------
    model = None
    if MODEL_VERSION == 1:
        model = XbD_FirstVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif MODEL_VERSION == 2:
        model = XbD_SecondVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif MODEL_VERSION == 3:
        model = XbD_ThirdVersion(num_classes=args.NUM_CLASSES, N=N, **shared_kwargs).to(device)
    elif MODEL_VERSION == 4:
        model = FrameMemoryTransformer(num_classes=args.NUM_CLASSES, memory_size=args.SEQ_LEN).to(device)
    else:
        raise ValueError(f"Invalid MODEL_VERSION: {MODEL_VERSION}. Must be 1, 2, 3, or 4.")
        
    criterion = ego_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # ----------------------------
    # Create output directory
    # ----------------------------
    output_dir = f"{ROOT}XbD/results/version{MODEL_VERSION}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------
    # Training
    # ----------------------------
    train(model, MODEL_VERSION, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs, patience, actual_input_len)

    # ----------------------------
    # Final Model Save
    # ----------------------------
    final_model_path = os.path.join(output_dir, "last_model_weights.pth")
    save_model_weights(model, final_model_path)
    print("--- Training Complete ---")


if __name__ == "__main__":
    main()
