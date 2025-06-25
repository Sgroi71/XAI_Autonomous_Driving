import torch
from torch.utils.data import DataLoader
import os
import sys
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import classification_report, f1_score

concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']

ROOT = '/home/jovyan/python/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/jovyan/nfs/lsgroi/'
sys.path.append("/home/jovyan/python/XAI_Autonomous_Driving/")
from XbD.data.dataset_prediction import VideoDataset
ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_stateless(model, dataloader ):
    X=[]
    y=[]
    with torch.no_grad():
        for batch in dataloader:
            labels_tensor = batch["labels"]
            ego_targets = batch["ego_labels"]

            labels_tensor = labels_tensor.squeeze(0)        # shape: (SEQ_LEN, N, 41)
            ego_targets = ego_targets.squeeze(0)  # shape: (SEQ_LEN,)

            
            for t in range(labels_tensor.shape[0]):
                if ego_targets[t].item() == -1:
                    continue  # salta frame non annotati
                X.append(labels_tensor[t].flatten().numpy())
                y.append(ego_targets[t].item()) 
        
            
    pred = model.predict(X)


    classification_report_dt = classification_report(y, pred, target_names=ego_actions_name)
    return classification_report_dt

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()
    maxdepth = 4
    model = DecisionTreeClassifier(max_depth=maxdepth)

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
    i=0
    for batch in dataloader_train:
        labels = batch['labels']      # shape: (1, SEQ_LEN, N, 41)
        ego_labels = batch['ego_labels']  # shape: (1, SEQ_LEN)
        labels = labels.squeeze(0)        # shape: (SEQ_LEN, N, 41)
        ego_labels = ego_labels.squeeze(0)  # shape: (SEQ_LEN,)
        
        for t in range(labels.shape[0]):
            if ego_labels[t].item() == -1:
                continue  # salta frame non annotati
            X.append(labels[t].flatten().numpy())
            y.append(ego_labels[t].item())
        # i+=1
        # if i==100:
        #     break
    X = np.array(X)
    y = np.array(y)

    model.fit(X, y)

    print("Model trained successfully.")
    cr=evaluate_stateless(model, dataloader_val)
    print(cr)

    feature_names = []
    for i in range(N):
        for cname in concept_names:
            feature_names.append(f"{cname}_{i}")

    plt.figure(figsize=(30, 15))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=ego_actions_name,label='none',impurity=False)
    plt.savefig(f'{ROOT}XbD/results_F1/versionDT/decision_tree_{maxdepth}.png', dpi=300)  

    plt.figure(figsize=(30, 15))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=ego_actions_name,label='none',impurity=False,node_ids=True)
    plt.savefig(f'{ROOT}XbD/results_F1/versionDT/decision_tree_{maxdepth}_ids.png', dpi=300) 



    ap_file_path = os.path.join(f'{ROOT}XbD/results_F1/versionDT', f"best_ap_strs_{maxdepth}.txt")
    with open(ap_file_path, "w") as f:
        f.write(cr)

    

    
if __name__ == "__main__":
    main()
