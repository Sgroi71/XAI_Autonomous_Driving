import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/jovyan/python/XAI_Autonomous_Driving/")
from XbD.models.second_version import XbD_SecondVersion
from XbD.data.dataset_prediction import VideoDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF


ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']
ROOT = '/home/jovyan/python/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/jovyan/nfs/lsgroi/'
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

def visualize_topk_attended(images, boxes, attentions, preds, top_k=5):
    """
    Visualize each image with the top-k most attended boxes and their attention values.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF

    B, T, C, H, W = images.shape
    _, _, N, _ = boxes.shape  # boxes: (B, T, N, 4)

    for b in range(B):
        for t in range(T):
            image_tensor = images[b, t]  # (C, H, W)
            image = TF.to_pil_image(image_tensor.cpu())

            attention = attentions[b, t]  # (N,)
            box_set = boxes[b, t]         # (N, 4)
            pred_t = preds[b, t]         # (,)  # Predicted class for this time step

            # Get top-k indices (handle if N < top_k)
            topk = min(top_k, box_set.shape[0])
            topk_scores, topk_indices = torch.topk(attention, k=topk)

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices)):
                box = box_set[idx].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1

                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Add attention score label
                ax.text(
                    x1, y1 - 5, f"{score.item():.3f}",
                    color='white', fontsize=10, backgroundcolor='red'
                )

            ax.set_title(f"Batch {b}, Time {t} - Top {topk} Attended Boxes - Predicted class: {ego_actions_name[pred_t]}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()

    # concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
    # ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

    version = '5d40f1af'  # specify the version of the model to load
    model = load_model_weights(
        XbD_SecondVersion,
        f"{ROOT}XbD/results_F1/version2/grid_{version}/best_model_v2_weights.pth",
        get_device(),
        num_classes=41,
        N=N
    )

    class Args:
        ANCHOR_TYPE = 'default'
        DATASET = 'road'
        SEQ_LEN = 1
        SUBSETS = ['val_3']
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
        PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadal-h3x3x3/detections-30-08-50')
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41

    args = Args()

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN,explaination=True)
    dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    batch = next(iter(dataloader))
    boxes = batch['boxes'].to(device)  # shape (B, T, N, 4)
    labels = batch['labels'].to(device)  # shape (B, T, N, num_classes)
    images = batch['images'].to(device)  # shape (B, T, C, H, W)
    logits, attentions = model(labels, return_attn=True, average_attn=True)

    preds = logits.argmax(dim=-1)
    visualize_topk_attended(images, boxes, attentions, preds, top_k=5)

if __name__ == "__main__":
    main()


