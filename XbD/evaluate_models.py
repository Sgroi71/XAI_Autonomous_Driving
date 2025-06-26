import json
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
from XbD.models.third_version import XbD_ThirdVersion
from XbD.models.second_version import XbD_SecondVersion
from XbD.models.first_version import XbD_FirstVersion
from XbD.data.dataset_prediction import VideoDataset
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import gc
import torch

ROOT = '/home/fabio/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/fabio/XAI_Autonomous_Driving/'
ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

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

def retrieve_best_configuration(filename):
    
    try:
        with open(filename, 'r') as file:
            configurations = json.load(file)
            return configurations[0]["cfg_id"], configurations[0]["params"]
    except Exception as e:
        print(f"Error retrieving best configuration: {e}")
        return None
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_model(model, dataloader, device):
    all_gt = []
    all_pred = []
    all_logits = []
    for batch in dataloader:
        ego_label_gt = batch['ego_labels']
        with torch.no_grad():
            # Move batch to device
            model.eval()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Forward pass with attention outputs, passing only batch["labels"]
            logits = model(batch["labels"])

        # logits: [B, T, num_classes], ego_label_gt: [B, T]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        gts = ego_label_gt.cpu().numpy()
        mask = (gts != -1)
        # Only keep logits where mask is True
        masked_logits = logits.cpu().numpy()[mask]
        all_logits.append(masked_logits)
        all_pred.extend(preds[mask].reshape(-1))
        all_gt.extend(gts[mask].reshape(-1))

    output_lines = []
    output_lines.append(f"Final classification report for model {model.__class__.__name__}:\n")
    output_lines.append(classification_report(
        all_gt, all_pred,
        labels=list(range(len(ego_actions_name))),
        target_names=ego_actions_name,
        zero_division=0
    ))
    mf1 = f1_score(all_gt, all_pred, average='macro', zero_division=0)
    wf1 = f1_score(all_gt, all_pred, average='weighted', zero_division=0)
    output_lines.append(f"Macro F1 score: {mf1:.4f}\n")
    output_lines.append(f"Weighted F1 score: {wf1:.4f}\n")
    all_gt_np = np.array(all_gt)
    all_logits_np = np.vstack(all_logits)  # shape: (n, 7)
    print("all_gt_np shape:", all_gt_np.shape)
    print("all_logits_np shape:", all_logits_np.shape)
    mAP, ap_all, ap_strs = evaluate_ego(all_gt_np, all_logits_np, ego_actions_name)
    output_lines.append(f"mAP: {mAP:.4f}\n")
    output_lines.append("Per class AP:\n")
    for ap_str in ap_strs:
        output_lines.append(f"{ap_str}\n")
    output_lines.append("Ap all: " + str([float(x) for x in ap_all]) + "\n")

    with open("output.txt", "a") as f:
        for line in output_lines:
            f.write(line)
        f.write('\n\n\n')

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()
    print("Working on device :", device)
    

    models = []

    conf_id, params = retrieve_best_configuration(f"{ROOT}XbD/results_F1/version3/grid_results.json")
    print("Configuration ID:", conf_id)
    print("Parameters:", params)

    models.append(load_model_weights(
        XbD_ThirdVersion,
        f"{ROOT}XbD/results_F1/version3/grid_{conf_id}/best_model_v3_weights.pth",
        get_device(),
        num_classes=41,
        d_model=params.get("d_model", 64),
        nhead_det=params.get("nhead", 2),
        nhead_time=params.get("nhead", 2),
        num_layers_time=params.get("num_layers", 1),
        num_layers_det=params.get("num_layers", 1),
        N=N
    ))

    conf_id, params = retrieve_best_configuration(f"{ROOT}XbD/results_F1/version2/grid_results.json")
    print("Configuration ID:", conf_id)
    print("Parameters:", params)
    models.append(load_model_weights(
        XbD_SecondVersion,
        f"{ROOT}XbD/results_F1/version2/grid_{conf_id}/best_model_v2_weights.pth",
        get_device(),
        num_classes=41,
        d_model=params.get("d_model", 64),
        nhead=params.get("nhead", 2),
        num_layers=params.get("num_layers", 1),
        N=N
    ))

    models.append(load_model_weights(
        XbD_FirstVersion,
        f"{ROOT}XbD/results_F1/version1/best_model_v1_weights.pth",
        get_device(),
        num_classes=41,
        N=N
    ))
    print("Models loaded successfully.")

    class Args:
        ANCHOR_TYPE = 'default'
        DATASET = 'road'
        SEQ_LEN = 8
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        SUBSETS = ['val_3']
        DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
        PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/detections-30-08-50')
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41

    args = Args()

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN, explaination=True)
    print("Created dataset with {} samples".format(len(dataset_val)))
    dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )
    print("Created dataloader with {} samples".format(len(dataloader)))

    for model in models:
        evaluate_model(model, dataloader, device)

main()