import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm  # for progress bar

from data.fake_dataset import FakeDataset
from models.first_version import XbD_FirstVersion
from models.second_version import XbD_SecondVersion
from models.third_version import XbD_ThirdVersion
from utils_loss import ego_loss
from data.dataset_prediction import VideoDataset
import numpy as np
import os
import matplotlib.pyplot as plt

ROOT= '/home/jovyan/python/XAI_Autonomous_Driving/'
ROOT_DATA= '/home/jovyan/nfs/lsgroi/'
model_version = 2

def evaluate_ego(gts, dets, classes):
    ap_strs = []
    num_frames = gts.shape[0]
    print('Evaluating for ' + str(num_frames) + ' frames')
    
    if num_frames<1:
        return 0, [0, 0], ['no gts present','no gts present']

    ap_all = []
    sap = 0.0
    for cls_ind, class_name in enumerate(classes):
        scores = dets[:, cls_ind]
        istp = np.zeros_like(gts)
        istp[gts == cls_ind] = 1
        det_count = num_frames
        num_postives = np.sum(istp)
        cls_ap = get_class_ap_from_scores(scores, istp, num_postives)
        ap_all.append(cls_ap)
        sap += cls_ap
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)
    
    mAP = sap/len(classes)
    ap_strs.append('FRAME Mean AP:: {:0.2f}'.format(mAP))
    
    return mAP, ap_all, ap_strs

def get_class_ap_from_scores(scores, istp, num_postives):
    # num_postives = np.sum(istp)
    if num_postives < 1:
        num_postives = 1
    argsort_scores = np.argsort(-scores)  # sort in descending order
    istp = istp[argsort_scores]  # reorder istp's on score sorting
    fp = np.cumsum(istp == 0)  # get false positives
    tp = np.cumsum(istp == 1)  # get  true positives
    fp = fp.astype(np.float64)
    tp = tp.astype(np.float64)
    recall = tp / float(num_postives)  # compute recall
    # compute precision
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # compute average precision using voc2007 metric
    cls_ap = voc_ap(recall, precision)
    return cls_ap

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap*100

def get_device():
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

        # Flatten both time and batch dims to compute loss:
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


def train(model, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs, patience=10):
    best_mAP = -float('inf')
    epochs_no_improve = 0
    best_model_path = f"{ROOT}XbD/results/version{model_version}/best_model_weights.pth"

    train_losses = []
    val_losses = []
    mAPs=[]

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_one_epoch(model, dataloader_train, criterion, optimizer, device)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}")

        print("Evaluating on validation set...")
        mAP, avg_val_loss, ap_strs = evaluate(model, dataloader_val, device, criterion=criterion)
        mAPs.append(mAP)
        val_losses.append(avg_val_loss)
        print(f"Validation mAP: {mAP:.4f}" if mAP is not None else "Validation mAP: None")
        print(f"Validation Loss: {avg_val_loss:.4f}" if avg_val_loss is not None else "Validation Loss: None")

        if mAP is not None and mAP > best_mAP:
            best_mAP = mAP
            epochs_no_improve = 0
            save_model_weights(model, best_model_path)
            # Save AP strings
            ap_file_path = f"{ROOT}XbD/results/version{model_version}/best_ap_strs.txt"
            with open(ap_file_path, "w") as f:
                for line in ap_strs:
                    f.write(line + "\n")
            print(f"New best mAP: {best_mAP:.4f} - model saved.")
            print(f"AP strings saved to {ap_file_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in mAP for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        print("-" * 50)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"{ROOT}XbD/results/version{model_version}/loss_curve.png"
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")


    #plot mAP
    plt.figure(figsize=(10, 5))
    plt.plot(mAPs, label='Val mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"{ROOT}XbD/results/version{model_version}/mAP_plot.png"
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_gts = []
    all_dets = []
    total_loss = 0.0
    num_batches = 0
    classes = [str(i) for i in range(7)]  # Assuming 7 ego classes

    with torch.no_grad():
        for batch in dataloader:
            labels_tensor = batch["labels"].to(device)
            ego_targets = batch["ego_labels"].to(device)  # (B, T)
            logits = model(labels_tensor)                 # (B, T, 7)
            activation = torch.nn.Sigmoid().cuda()
            preds = activation(logits)         

            B, T, _ = logits.shape
            all_gts.append(ego_targets.view(-1).cpu().numpy())
            all_dets.append(preds.view(-1, 7).cpu().numpy())

            if criterion:
                logits_flat = logits.view(B * T, 7)
                targets_flat = ego_targets.view(B * T)
                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1

    if not all_gts or not all_dets:
        print("No data for evaluation.")
        return None, None

    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)

    mAP, ap_all, ap_strs = evaluate_ego(gts, dets, classes)
    print("\nEvaluation Results:")
    for s in ap_strs:
        print(s)

    avg_val_loss = total_loss / num_batches if num_batches > 0 else None
    return mAP, avg_val_loss, ap_strs


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


def save_model_weights(model: nn.Module, path: str):
    """
    Save the given model's state_dict to the specified path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model_weights(model_class, checkpoint_path: str, device, **model_kwargs):
    """
    Instantiate a fresh model of type `model_class` using `model_kwargs`,
    load state_dict from `checkpoint_path`, and move to `device`.
    Returns the loaded model.
    """
    model = model_class(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {checkpoint_path}")
    return model


def main():
    # ----------------------------
    # Hyperparameters & Settings
    # ----------------------------
    #length = 1_000_000       # number of samples in FakeDataset
    T = 8                    # temporal dimension
    N = 10                    # number of objects per time step      
    batch_size = 1024
    num_epochs = 500
    learning_rate = 1e-3
    patience = 200

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
        DATASET = 'road'  # o 'ucf24', 'ava'
        SUBSETS = ['train_3']
        SEQ_LEN = T
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = f'{ROOT_DATA}dataset/'
        PREDICTION_ROOT = f'{ROOT}road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadal-h3x3x3/detections-30-08-50'
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41

    args = Args()
    # Durante il gen_dets usa skip_step = SEQ_LEN -  2
    # Durante il training usa skip_step = SEQ_LEN
    # Durante il validation usa skip_step = SEQ_LEN*8   ma a noi non ci interessa
    dataset_train = VideoDataset(args, train=True, input_type='rgb', transform=None, skip_step=args.SEQ_LEN, full_test=False)

    args.SUBSETS=['val_3']

    dataset_val = VideoDataset(args, train=True, input_type='rgb', transform=None, skip_step=args.SEQ_LEN, full_test=False)

    # Crea il  train DataLoader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    # Crea il validation DataLoader
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )
    

    # ----------------------------
    # Model, Loss, Optimizer
    # ----------------------------
    if model_version<2:
        model = XbD_FirstVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    elif model_version>=2 and model_version<3:
        model = XbD_SecondVersion(num_classes=args.NUM_CLASSES, N=N).to(device)
    else:
        print ("model version error")
    criterion = ego_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4  # a small L2 penalty on weights
    )

    # ----------------------------
    # Create output directory for saving results
    # ----------------------------
    output_dir = f"{ROOT}XbD/results/version{model_version}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # ----------------------------
    # Training
    # ----------------------------
    train(model, dataloader_train,dataloader_val, criterion, optimizer, device, num_epochs, patience)

    # ----------------------------
    # Save model weights
    # ----------------------------
    
    #checkpoint_path = f"{ROOT}XbD/results/version{model_version}/best_model_weights.pth"
    #save_model_weights(model, checkpoint_path)

    ########### Example of inference on a sample ###########

    # ----------------------------
    # Example: loading the model back
    # ----------------------------
    # loaded_model = load_model_weights(
    #     XbD_FirstVersion,
    #     checkpoint_path,
    #     device,
    #     num_classes=args.NUM_CLASSES,
    #     N=N
    # )

    # ----------------------------
    # Example Inference on a Sample using the loaded model
    # ----------------------------
    #_,_,_=evaluate(loaded_model, dataloader_val, device, criterion=criterion)


if __name__ == "__main__":
    main()

def evaluate_ego(gts, dets, classes):
    ap_strs = []
    num_frames = gts.shape[0]
    print('Evaluating for ' + str(num_frames) + ' frames')
    
    if num_frames<1:
        return 0, [0, 0], ['no gts present','no gts present']

    ap_all = []
    sap = 0.0
    for cls_ind, class_name in enumerate(classes):
        scores = dets[:, cls_ind]
        istp = np.zeros_like(gts)
        istp[gts == cls_ind] = 1
        det_count = num_frames
        num_postives = np.sum(istp)
        cls_ap = get_class_ap_from_scores(scores, istp, num_postives)
        ap_all.append(cls_ap)
        sap += cls_ap
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)
    
    mAP = sap/len(classes)
    ap_strs.append('FRAME Mean AP:: {:0.2f}'.format(mAP))
    
    return mAP, ap_all, ap_strs

def get_class_ap_from_scores(scores, istp, num_postives):
    # num_postives = np.sum(istp)
    if num_postives < 1:
        num_postives = 1
    argsort_scores = np.argsort(-scores)  # sort in descending order
    istp = istp[argsort_scores]  # reorder istp's on score sorting
    fp = np.cumsum(istp == 0)  # get false positives
    tp = np.cumsum(istp == 1)  # get  true positives
    fp = fp.astype(np.float64)
    tp = tp.astype(np.float64)
    recall = tp / float(num_postives)  # compute recall
    # compute precision
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # compute average precision using voc2007 metric
    cls_ap = voc_ap(recall, precision)
    return cls_ap

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap*100