import json
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
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
all_logit = []
for batch in dataloader:
    ego_label_gt = batch['ego_labels']
    ego_logit = batch['ego_pred'][0]
    ego_label_pred = ego_logit.argmax(dim=1)

    gt_flat = ego_label_gt.flatten().numpy()
    pred_flat = ego_label_pred.flatten().numpy()
    i=0
    for gt, pred in zip(gt_flat, pred_flat):
        if gt != -1:
            all_gt.append(gt)
            all_pred.append(pred)
            all_logit.append(ego_logit[i])
        i+=1

# Calculate classification report
report = classification_report(all_gt, all_pred, target_names=ego_actions_name)
print("Classification Report:\n", report)

# Calculate mean F1 (macro) and weighted F1
f1_macro = f1_score(all_gt, all_pred, average='macro')
f1_weighted = f1_score(all_gt, all_pred, average='weighted')
print(f"Mean F1 (macro): {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")

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


all_gt_np = np.array(all_gt)
all_logits_np = np.vstack(all_logit)  # shape: (n, 7)
print("all_gt_np shape:", all_gt_np.shape)
print("all_logits_np shape:", all_logits_np.shape)
mAP, ap_all, ap_strs = evaluate_ego(all_gt_np, all_logits_np, ego_actions_name)
print(f"mAP: {mAP:.4f}\n")
print("Per class AP:\n")
for ap_str in ap_strs:
    print(f"{ap_str}\n")
print("Ap all: " + str([float(x) for x in ap_all]) + "\n")