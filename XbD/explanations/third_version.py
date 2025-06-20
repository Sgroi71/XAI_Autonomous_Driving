import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
from XbD.models.third_version import XbD_ThirdVersion
from XbD.data.dataset_prediction import VideoDataset
import matplotlib.pyplot as plt
import numpy as np



ROOT = '/home/fabio/XAI_Autonomous_Driving/'
ROOT_DATA = '/home/fabio/XAI_Autonomous_Driving/'
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

def plot_attention_maps_for_detection(attn_maps_list, plot_title=None):
    """
    Plots detection attention maps for each layer.
    All heads for all layers are shown together in a grid for each batch/time step.
    attn_maps_list: list of tensors, each of shape (B, T, num_heads, N)
    """
    for b in range(attn_maps_list[0].shape[0]):  # batch
        for t in range(attn_maps_list[0].shape[1]):  # time
            num_layers = len(attn_maps_list)
            num_heads = attn_maps_list[0].shape[2]
            N = attn_maps_list[0].shape[3]
            fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))
            if plot_title:
                fig.suptitle(f"{plot_title} - Batch {b}, Time {t}", fontsize=14)
            for layer_idx, attn_maps in enumerate(attn_maps_list):
                attn_maps_np = attn_maps.detach().cpu().numpy()
                for h in range(num_heads):
                    ax = axes[layer_idx, h] if num_layers > 1 else axes[h]
                    im = ax.imshow(attn_maps_np[b, t, h][np.newaxis, :], aspect="auto", cmap="gist_heat")
                    if layer_idx == 0:
                        ax.set_title(f"Head {h+1}")
                    if h == 0:
                        ax.set_ylabel(f"Layer {layer_idx+1}")
                    ax.set_xticks(np.arange(N))
                    ax.set_xticklabels([f"Box {i}" for i in range(N)], rotation=90)
                    ax.set_yticks([0])
                    ax.set_yticklabels(["CLS"])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()
    print("Working on device :", device)

    concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
    ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

    model = load_model_weights(
        XbD_ThirdVersion,
        f"{ROOT}XbD/results/version3/best_model_weights.pth",
        get_device(),
        num_classes=41,
        N=N
    )

    class Args:
        ANCHOR_TYPE = 'default'
        DATASET = 'road'
        SEQ_LEN = 8
        SUBSETS = ['val_3']
        MIN_SEQ_STEP = 1
        MAX_SEQ_STEP = 1
        DATA_ROOT = os.path.join(ROOT_DATA, 'dataset/')
        PREDICTION_ROOT = os.path.join(ROOT, 'road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/detections-30-08-50')
        MAX_ANCHOR_BOXES = N
        NUM_CLASSES = 41

    args = Args()

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)
    print("Created dataset with {} samples".format(len(dataset_val)))
    dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )
    batch = next(iter(dataloader))
    print("Batch sampled from dataset:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.shape} on device {value.device}")
        else:
            print(f"{key}: {value}")

    with torch.no_grad():
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Forward pass with attention outputs, passing only batch["labels"]
        logits, attn_det, attn_time = model(
            batch["labels"],
            return_attn=True,
            average_heads=False
        )

    # Plot detection transformer attention maps
    print("Detection Transformer Attention Maps:")
    plot_attention_maps_for_detection(attn_det, plot_title="Detection Transformer")

    # Plot time transformer attention maps
    """ print("Time Transformer Attention Maps:")
    plot_attention_maps(None, attn_time, "Time transformer") """




if __name__ == "__main__":
    main()


