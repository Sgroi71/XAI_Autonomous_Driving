from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid  # Lightweight sweep utility

# --- Model Imports -----------------------------------------------------------
from models.first_version import XbD_FirstVersion
from models.second_version import XbD_SecondVersion
from models.third_version import XbD_ThirdVersion
from models.fourth_version import FrameMemoryTransformer

# --- Data & Loss -------------------------------------------------------------
from data.dataset_prediction import VideoDataset
from utils_loss import ego_loss

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
ROOT = "/home/jovyan/python/XAI_Autonomous_Driving/"
ROOT_DATA = "/home/jovyan/nfs/lsgroi/"

# Select model version --------------------------------------------------------
MODEL_VERSION: int = 2  # 1, 2, 3, or 4

# Enable / disable grid‑search (only relevant for v2/v3) ----------------------
ENABLE_GRID_SEARCH: bool = MODEL_VERSION in {2, 3}

# Quick training schedule for grid‑search -------------------------------------
SEARCH_EPOCHS: int = 200
SEARCH_PATIENCE: int = 25

# Parameter grids -------------------------------------------------------------
GRID_V2: Dict[str, List[Any]] = {
    "d_model": [32, 64],
    "nhead": [1, 2, 4],
    "num_layers": [1, 2],
    "dropout": [0.3, 0.1],
    "lr": [1e-5, 5e-5, 1e-4],  # optimiser learning‑rate
}

GRID_V3: Dict[str, List[Any]] = {
    "d_model": [32, 64, 128],          # shared across branches
    "nhead": [1, 2, 4],               # used for BOTH det & time streams
    "num_layers": [1, 2],          # used for BOTH det & time streams
    "dropout": [0.3, 0.1],              # single regularisation value
    "lr": [1e-5, 5e-5, 1e-4],            # optimiser learning‑rate
}

# -----------------------------------------------------------------------------
# Utility & evaluation functions (unchanged)
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_ego(gts: np.ndarray, dets: np.ndarray, classes: List[str]):
    """Evaluate ego‑motion prediction with mean AP (mAP)."""
    ap_strs = []
    num_frames = gts.shape[0]
    print(f"Evaluating for {num_frames} frames")

    if num_frames < 1:
        return 0, [0, 0], ["no gts present", "no gts present"]

    ap_all, sap = [], 0.0
    for cls_ind, class_name in enumerate(classes):
        scores = dets[:, cls_ind]
        istp = np.zeros_like(gts)
        istp[gts == cls_ind] = 1
        det_count = num_frames
        num_positives = np.sum(istp)
        cls_ap = get_class_ap_from_scores(scores, istp, num_positives)
        ap_all.append(cls_ap)
        sap += cls_ap
        ap_strs.append(f"{class_name} : {num_positives} : {det_count} : {cls_ap}")

    mAP = sap / len(classes)
    ap_strs.append(f"FRAME Mean AP:: {mAP:0.2f}")
    return mAP, ap_all, ap_strs


def get_class_ap_from_scores(scores, istp, num_positives):
    if num_positives < 1:
        num_positives = 1
    argsort_scores = np.argsort(-scores)
    istp = istp[argsort_scores]
    fp = np.cumsum(istp == 0).astype(np.float64)
    tp = np.cumsum(istp == 1).astype(np.float64)
    recall = tp / float(num_positives)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return voc_ap(recall, precision)


def voc_ap(rec, prec, use_07_metric: bool = False):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) else 0
            ap += p / 11.0
        return ap * 100
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) * 100


def save_model_weights(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model_weights(model_class, checkpoint_path: str, device: torch.device, **model_kwargs):
    model = model_class(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {checkpoint_path}")
    return model

# -----------------------------------------------------------------------------
# Training & evaluation (stateless versions 1‑3) ------------------------------
# -----------------------------------------------------------------------------

def train_one_epoch_stateless(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training (Stateless)", leave=False)
    for batch in loop:
        labels_tensor = batch["labels"].to(device)
        ego_targets = batch["ego_labels"].to(device)

        logits = model(labels_tensor)
        B, T, _ = logits.shape
        loss = criterion(logits.view(B * T, 7), ego_targets.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(dataloader)


def evaluate_stateless(model, dataloader, device, criterion=None):
    model.eval()
    classes = [str(i) for i in range(7)]
    all_gts, all_dets = [], []
    total_loss, num_batches = 0.0, 0
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
                total_loss += criterion(logits.view(B * T, 7), ego_targets.view(B * T)).item()
                num_batches += 1
    if not all_gts:
        return None, None, ["No data for evaluation."]
    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)
    mAP, _, ap_strs = evaluate_ego(gts, dets, classes)
    avg_loss = total_loss / num_batches if num_batches else None
    return mAP, avg_loss, ap_strs

# -----------------------------------------------------------------------------
# Training & evaluation (memory version 4) ------------------------------------
# -----------------------------------------------------------------------------

def train_one_epoch_memory(model, dataloader, criterion, optimizer, device, actual_seq_len):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training (Memory)", leave=False)
    for batch in loop:
        B, T, _, _ = batch["labels"].shape
        num_slices = T // actual_seq_len
        prev_memory, total_loss_for_clip = None, 0.0
        optimizer.zero_grad()
        for i in range(num_slices):
            s, e = i * actual_seq_len, (i + 1) * actual_seq_len
            labels_slice = batch["labels"][:, s:e].to(device)
            targets_slice = batch["ego_labels"][:, s:e].to(device)
            if prev_memory is not None:
                prev_memory = prev_memory.detach().to(device)
            logits, prev_memory = model(labels_slice, prev_memory)
            loss = criterion(logits.view(-1, 7), targets_slice.view(-1))
            total_loss_for_clip += loss
        total_loss_for_clip.backward()
        optimizer.step()
        running_loss += total_loss_for_clip.item()
        loop.set_postfix(loss=total_loss_for_clip.item())
    return running_loss / len(dataloader)


def evaluate_memory(model, dataloader, device, criterion=None, actual_seq_len: int = 8):
    model.eval()
    classes = [str(i) for i in range(7)]
    all_gts, all_dets = [], []
    total_loss, num_clips = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            B, T, _, _ = batch["labels"].shape
            num_slices = T // actual_seq_len
            prev_memory = None
            for i in range(num_slices):
                s, e = i * actual_seq_len, (i + 1) * actual_seq_len
                labels_slice = batch["labels"][:, s:e].to(device)
                targets_slice = batch["ego_labels"][:, s:e].to(device)
                if prev_memory is not None:
                    prev_memory = prev_memory.to(device)
                logits, prev_memory = model(labels_slice, prev_memory)
                preds = torch.sigmoid(logits)
                all_gts.append(targets_slice.view(-1).cpu().numpy())
                all_dets.append(preds.view(-1, 7).cpu().numpy())
                if criterion:
                    total_loss += criterion(logits.view(-1, 7), targets_slice.view(-1)).item()
            num_clips += 1
    if not all_gts:
        return None, None, ["No data for evaluation."]
    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)
    mAP, _, ap_strs = evaluate_ego(gts, dets, classes)
    avg_loss = total_loss / (num_clips * num_slices) if num_clips else None
    return mAP, avg_loss, ap_strs

# -----------------------------------------------------------------------------
# Training orchestrator (unchanged except for optional output_dir argument)
# -----------------------------------------------------------------------------

def train(model, model_version: int, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs: int, patience: int, actual_seq_len: int | None = None, output_dir: str | None = None):
    output_dir = output_dir or f"{ROOT}XbD/results/version{model_version}"
    os.makedirs(output_dir, exist_ok=True)
    best_mAP, epochs_no_improve = -float("inf"), 0
    best_model_path = os.path.join(output_dir, f"best_model_v{model_version}_weights.pth")
    train_losses, val_losses, mAPs = [], [], []
    for epoch in range(1, num_epochs + 1):
        if model_version == 4:
            avg_train_loss = train_one_epoch_memory(model, dataloader_train, criterion, optimizer, device, actual_seq_len)
        else:
            avg_train_loss = train_one_epoch_stateless(model, dataloader_train, criterion, optimizer, device)
        train_losses.append(avg_train_loss)

        # Validation
        if model_version == 4:
            mAP, avg_val_loss, _ = evaluate_memory(model, dataloader_val, device, criterion, actual_seq_len)
        else:
            mAP, avg_val_loss, _ = evaluate_stateless(model, dataloader_val, device, criterion)
        mAPs.append(mAP)
        val_losses.append(avg_val_loss)

        # Check improvement
        if mAP is not None and mAP > best_mAP:
            best_mAP, epochs_no_improve = mAP, 0
            save_model_weights(model, best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping (no mAP improvement for {patience} epochs)")
            break

    # Curves -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
    plt.title(f"Loss Curve (v{model_version})")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(mAPs, label="Val mAP")
    plt.xlabel("Epoch"); plt.ylabel("mAP"); plt.grid(True); plt.legend()
    plt.title(f"Validation mAP (v{model_version})")
    plt.savefig(os.path.join(output_dir, "mAP_plot.png"))
    plt.close()

    return best_mAP  # for grid‑search

# -----------------------------------------------------------------------------
# Helper functions for grid‑search -------------------------------------------
# -----------------------------------------------------------------------------

def cfg_hash(cfg: Dict[str, Any]) -> str:
    """Stable eight‑char hash for folder names."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def build_model(model_version: int, N: int, num_classes: int, device: torch.device, cfg: Dict[str, Any]):
    if model_version == 2:
        model = XbD_SecondVersion(N=N, num_classes=num_classes, **{k: cfg[k] for k in ["d_model", "nhead", "num_layers", "dropout"]})
    elif model_version == 3:
        # Shared heads/layers -> apply to both detector & temporal branches
        shared_kwargs = {
            "d_model": cfg["d_model"],
            "nhead_det": cfg["nhead"],
            "num_layers_det": cfg["num_layers"],
            "nhead_time": cfg["nhead"],
            "num_layers_time": cfg["num_layers"],
            "dropout": cfg["dropout"],
        }
        model = XbD_ThirdVersion(N=N, num_classes=num_classes, **shared_kwargs)
    else:
        raise ValueError("Grid‑search only implemented for versions 2 and 3.")
    return model.to(device)

# -----------------------------------------------------------------------------
# Main dataset/loader factory (shared) ---------------------------------------
# -----------------------------------------------------------------------------

def make_dataloaders(T: int, N: int, batch_size: int, actual_input_len: int | None, train_subsets: List[str], val_subsets: List[str]):
    class Args:
        ANCHOR_TYPE = "default"
        DATASET = "road"
        SEQ_LEN = T
        SUBSETS = train_subsets
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = os.path.join(ROOT_DATA, "dataset/")
        PREDICTION_ROOT = os.path.join(ROOT, "road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadal-h3x3x3/detections-30-08-50")
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41
        if MODEL_VERSION == 4:
            ACTUAL_SEQ_LEN = actual_input_len
    args = Args()

    dataset_train = VideoDataset(args, train=True, skip_step=args.SEQ_LEN)
    args.SUBSETS = val_subsets
    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)

    device = get_device()
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=device.type == "cuda")
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=device.type == "cuda")
    return args, dataloader_train, dataloader_val

# -----------------------------------------------------------------------------
# Grid‑search routine ---------------------------------------------------------
# -----------------------------------------------------------------------------

def run_grid_search():
    """Run exhaustive grid‑search for the selected model version."""
    print(f"*** Grid‑search for Version {MODEL_VERSION} ***")

    # Default hyper‑params (same as original main()) ---------------------
    T, batch_size, actual_input_len = (8, 1024, None) if MODEL_VERSION != 4 else (48, 256, 8)
    N = 10
    param_grid = GRID_V2 if MODEL_VERSION == 2 else GRID_V3
    param_iter = list(ParameterGrid(param_grid))
    print(f"Total configurations: {len(param_iter)}\n")

    # Data loaders only depend on T/N/batch_size (constant across cfgs)
    train_subsets, val_subsets = ["train_3"], ["val_3"]
    args, dl_train, dl_val = make_dataloaders(T, N, batch_size, actual_input_len, train_subsets, val_subsets)
    device = get_device()

    # Sweep --------------------------------------------------------------
    results = []
    for cfg_idx, cfg in enumerate(param_iter, 1):
        lr = float(cfg.pop("lr"))  # optimiser param – not for model ctor
        cfg_id = cfg_hash(cfg | {"lr": lr})
        print(f"\n>>> [{cfg_idx}/{len(param_iter)}] Config {cfg_id}: {cfg}, lr={lr}")

        model = build_model(MODEL_VERSION, N=N, num_classes=args.NUM_CLASSES, device=device, cfg=cfg)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = ego_loss

        out_dir = f"{ROOT}XbD/results/version{MODEL_VERSION}/grid_{cfg_id}"
        best_map = train(model, MODEL_VERSION, dl_train, dl_val, criterion, optimizer, device, SEARCH_EPOCHS, SEARCH_PATIENCE, actual_input_len, output_dir=out_dir)
        results.append({"cfg_id": cfg_id, "params": cfg, "lr": lr, "mAP": best_map})

    # Save + print summary ----------------------------------------------
    results.sort(key=lambda d: d["mAP"], reverse=True)
    summary_path = f"{ROOT}XbD/results/version{MODEL_VERSION}/grid_results.json"
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n=== Grid‑search complete ===")
    for rank, res in enumerate(results[:10], 1):
        print(f"#{rank:2d}  mAP={res['mAP']:.2f} | lr={res['lr']:.0e} | {res['params']}")

# -----------------------------------------------------------------------------
# Fallback: original single‑run pipeline -------------------------------------
# -----------------------------------------------------------------------------

def single_run():
    """Run a single training session using defaults (no grid‑search)."""
    print(f"--- Running Experiment for Model Version {MODEL_VERSION} (single‑run) ---")

    # Hyper‑parameters (original defaults) --------------------------------
    T, batch_size, actual_input_len = (8, 1024, None)
    if MODEL_VERSION == 4:
        T, actual_input_len, batch_size = 48, 8, 256
    N, num_epochs, patience, learning_rate = 10, 500, 100, 1e-5

    # Data ---------------------------------------------------------------
    train_subsets, val_subsets = ["train_3"], ["val_3"]
    args, dl_train, dl_val = make_dataloaders(T, N, batch_size, actual_input_len, train_subsets, val_subsets)
    device = get_device()

    # Model --------------------------------------------------------------
    if MODEL_VERSION == 1:
        model = XbD_FirstVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif MODEL_VERSION == 2:
        model = XbD_SecondVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif MODEL_VERSION == 3:
        model = XbD_ThirdVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif MODEL_VERSION == 4:
        model = FrameMemoryTransformer(num_classes=args.NUM_CLASSES, memory_size=T).to(device)
    else:
        raise ValueError("Invalid MODEL_VERSION")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = ego_loss

    # Output directory ---------------------------------------------------
    out_dir = f"{ROOT}XbD/results/version{MODEL_VERSION}"
    os.makedirs(out_dir, exist_ok=True)

    # Training -----------------------------------------------------------
    train(model, MODEL_VERSION, dl_train, dl_val, criterion, optimizer, device, num_epochs, patience, actual_input_len, out_dir)

    # Final save ---------------------------------------------------------
    save_model_weights(model, os.path.join(out_dir, f"last_model_v{MODEL_VERSION}.pth"))

    print("--- Training complete ---")

# -----------------------------------------------------------------------------
# Entry point -----------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if ENABLE_GRID_SEARCH:
        run_grid_search()
    else:
        single_run()