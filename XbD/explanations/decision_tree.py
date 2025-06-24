
from typing import List
import torch
from torch.utils.data import DataLoader
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from XbD.data.dataset_prediction import VideoDataset
import numpy as np

from sklearn.metrics import f1_score

ROOT = '/home/jovyan/python/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/jovyan/nfs/lsgroi/'
sys.path.append("/home/jovyan/python/XAI_Autonomous_Driving/")
from XbD.data.dataset_prediction import VideoDataset
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def evaluate_stateless(model, dataloader, device, criterion=None):
    """
    Evaluates stateless models (versions 1, 2, 3).
    """
    model.eval()
    all_gts, all_dets = [], []
    total_loss, num_batches = 0.0, 0
    classes = [str(i) for i in range(7)]
    total_correct = 0
    total_samples = 0
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            labels_tensor = batch["labels"].to(device)
            ego_targets = batch["ego_labels"].to(device)
            
            logits = model(labels_tensor)
            preds = torch.sigmoid(logits)

            pred = torch.argmax(logits, dim=-1)  # shape: B×T
            total_correct += (pred == ego_targets).sum().item()
            total_samples += pred.numel()

            B, T, _ = logits.shape
            all_gts.append(ego_targets.view(-1).cpu().numpy())
            all_dets.append(preds.view(-1, 7).cpu().numpy())
            all_preds.append(pred.view(-1).cpu().numpy())

            if criterion:
                loss = criterion(logits.view(B * T, 7), ego_targets.view(B * T))
                total_loss += loss.item()
                num_batches += 1

    if not all_gts:
        return None, None, ["No data for evaluation."], None, None

    gts = np.concatenate(all_gts, axis=0)
    dets = np.concatenate(all_dets, axis=0)
    preds_flat = np.concatenate(all_preds, axis=0)
    
    mAP, ap_all, ap_strs = evaluate_ego(gts, dets, classes)
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    f1 = f1_score(gts, preds_flat, average="weighted")
    return mAP, avg_loss, ap_strs, accuracy, f1

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()

    concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
    ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

    model = DecisionTreeClassifier()

    class Args:
        ANCHOR_TYPE = 'default'
        DATASET = 'road'
        SEQ_LEN = 1
        SUBSETS = ['train_3']
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
        PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadal-h3x3x3/detections-30-08-50')
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41

    args = Args()

    dataset_train = VideoDataset(args, train=True, skip_step=args.SEQ_LEN)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    args.SUBSETS = ['val_3']

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )
    
    X = []
    y = []
    for batch in DataLoader(dataset_train, batch_size=1):
        labels = batch['labels']      # shape: (1, SEQ_LEN, N, 41)
        ego_labels = batch['ego_labels']  # shape: (1, SEQ_LEN)
        labels = labels.squeeze(0)        # shape: (SEQ_LEN, N, 41)
        ego_labels = ego_labels.squeeze(0)  # shape: (SEQ_LEN,)

        for t in range(labels.shape[0]):
            if ego_labels[t].item() == -1:
                continue  # salta frame non annotati
            X.append(labels[t].flatten().numpy())
            y.append(ego_labels[t].item())

    X = np.array(X)
    y = np.array(y)

    model.fit(X, y)


    print("Model trained successfully.")
    # mAP, avg_loss, ap_strs, accuracy, f1 = evaluate_stateless(model, dataloader, device)

    # print(f"mAP: {mAP:.2f}")
    # print(f"Average Loss: {avg_loss:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(ap_strs)

    
if __name__ == "__main__":
    main()
