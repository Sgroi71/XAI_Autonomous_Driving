import json
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

def retrieve_best_configuration(filename):
    
    try:
        with open(filename, 'r') as file:
            configurations = json.load(file)
            return configurations[0]["cfg_id"], configurations[0]["params"]
    except Exception as e:
        print(f"Error retrieving best configuration: {e}")
        return None

def plot_attention_maps_for_detection(attn_maps_list, plot_title=None):
    """
    Plots all detection attention maps for each layer, head, batch, and time step together in a single image.
    Each subplot corresponds to (layer, head, batch, time).
    attn_maps_list: list of tensors, each of shape (B, T, num_heads, N) or (B, T, N) if num_heads == 1.
    """
    num_layers = len(attn_maps_list)
    sample_shape = attn_maps_list[0].shape
    if len(sample_shape) == 4:
        B, T, num_heads, N = sample_shape
        head_axis = 2
    elif len(sample_shape) == 3:
        B, T, N = sample_shape
        num_heads = 1
        head_axis = None
    else:
        raise ValueError(f"Unexpected attention map shape: {sample_shape}")

    total_plots = num_layers * num_heads * B * T
    cols = min(8, total_plots)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    plot_idx = 0
    for layer_idx, attn_maps in enumerate(attn_maps_list):
        attn_maps_np = attn_maps.detach().cpu().numpy()
        for b in range(B):
            for t in range(T):
                for h in range(num_heads):
                    ax = axes[plot_idx]
                    if head_axis is not None:
                        attn = attn_maps_np[b, t, h]
                        title = f"L{layer_idx+1} H{h+1} B{b} T{t}"
                    else:
                        attn = attn_maps_np[b, t]
                        title = f"L{layer_idx+1} B{b} T{t} (heads merged)"
                    im = ax.imshow(attn[np.newaxis, :], aspect="auto", cmap="gist_heat")
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks(np.arange(N))
                    ax.set_xticklabels([f"Box {i}" for i in range(N)], rotation=90, fontsize=6)
                    ax.set_yticks([0])
                    ax.set_yticklabels(["CLS"], fontsize=6)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_attention_maps_for_time(attn_maps_list, plot_title=None, frame_labels=None):
    """
    Plots all time transformer attention maps for each layer, head, and batch.
    Each attention map is (num_frames, num_frames) for a given (layer, head, batch).
    attn_maps_list: list of tensors, each of shape (B, num_heads, num_frames, num_frames)
    or (B, num_frames, num_frames) if num_heads == 1.
    """
    num_layers = len(attn_maps_list)
    # Determine shape and handle num_heads==1 case
    sample_shape = attn_maps_list[0].shape
    if len(sample_shape) == 4:
        B = sample_shape[0]
        num_heads = sample_shape[1]
        num_frames = sample_shape[2]
        head_axis = 1
    elif len(sample_shape) == 3:
        B = sample_shape[0]
        num_heads = 1
        num_frames = sample_shape[1]
        head_axis = None
    else:
        raise ValueError("Unexpected attention map shape: {}".format(sample_shape))

    total_plots = num_layers * num_heads * B
    cols = min(8, total_plots)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    plot_idx = 0
    for layer_idx, attn_maps in enumerate(attn_maps_list):
        attn_maps_np = attn_maps.detach().cpu().numpy()
        for b in range(B):
            for h in range(num_heads):
                ax = axes[plot_idx]
                if head_axis is not None:
                    attn = attn_maps_np[b, h]
                    title = f"L{layer_idx+1} H{h+1} B{b}"
                else:
                    attn = attn_maps_np[b]
                    title = f"L{layer_idx+1} B{b} (heads merged)"
                im = ax.imshow(attn, aspect="auto", cmap="viridis")
                ax.set_title(title, fontsize=8)
                ax.set_xticks(np.arange(num_frames))
                ax.set_yticks(np.arange(num_frames))
                if frame_labels is not None:
                    ax.set_xticklabels(frame_labels, rotation=90, fontsize=6)
                    ax.set_yticklabels(frame_labels, fontsize=6)
                else:
                    ax.set_xticklabels([f"F{i}" for i in range(num_frames)], rotation=90, fontsize=6)
                    ax.set_yticklabels([f"F{i}" for i in range(num_frames)], fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()
    print("Working on device :", device)

    concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
    ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']
    
    conf_id, params = retrieve_best_configuration(f"{ROOT}XbD/results_F1/version3/grid_results.json")
    print("Configuration ID:", conf_id)
    print("Parameters:", params)
    model = load_model_weights(
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
            average_heads=True
        )

    # Plot detection transformer attention maps
    print("Detection Transformer Attention Maps:")
    plot_attention_maps_for_detection(attn_det, plot_title="Detection Transformer")

    # Plot time transformer attention maps
    print("Time Transformer Attention Maps:")
    plot_attention_maps_for_time(attn_time, "Time transformer")

if __name__ == "__main__":
    main()


