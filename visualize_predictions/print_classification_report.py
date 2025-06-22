import json
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
from XbD.models.third_version import XbD_ThirdVersion
from XbD.data.dataset_prediction import VideoDataset
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

print(f"Current working directory: {os.getcwd()}")
ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

ROOT = '/home/fabio/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/fabio/XAI_Autonomous_Driving/'
class Args:
    ANCHOR_TYPE = 'default'
    DATASET = 'road'
    SEQ_LEN = 8
    MIN_SEQ_STEP = 1
    MAX_SEQ_STEP = 1
    SUBSETS = ['val_3']
    DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
    PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/detections-30-08-50')
    MAX_ANCHOR_BOXES = 10
    NUM_CLASSES = 41

args = Args()

dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)
print("Created dataset with {} samples".format(len(dataset_val)))
dataloader = DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=4,       # adjust num_workers as needed
    pin_memory=True
)

all_pred = []
all_gt = []
for batch in dataloader:
    ego_label_gt = batch['ego_labels']
    ego_label_pred = batch['ego_pred'].argmax(dim=2)

    gt_flat = ego_label_gt.flatten().numpy()
    pred_flat = ego_label_pred.flatten().numpy()
    for gt, pred in zip(gt_flat, pred_flat):
        if gt != -1:
            all_gt.append(gt)
            all_pred.append(pred)

# Calculate classification report
report = classification_report(all_gt, all_pred, target_names=ego_actions_name)
print("Classification Report:\n", report)

# Calculate mean F1 (macro) and weighted F1
f1_macro = f1_score(all_gt, all_pred, average='macro')
f1_weighted = f1_score(all_gt, all_pred, average='weighted')
print(f"Mean F1 (macro): {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")