import json
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
from XbD.models.third_version import XbD_ThirdVersion
from XbD.data.dataset_prediction_bounded_test import VideoDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

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

def get_rollout_attention(attentions):
    """
    Given a list of attention maps (one per layer), already aggregated over heads,
    compute the rollout (cumulative product over layers).
    Each attention in the list is of shape (B, T, N) or (B, N, N), etc.
    Returns: Tensor of shape (B, T, N) or (B, N, N) depending on input.
    """

    # assert that all attentions have 3 dimensions
    if not all(len(att.shape) == 3 for att in attentions):
        raise ValueError("All attention maps must have 3 dimensions (B, T, N) or (B, N, N), remember to average on the heads")

    rollout = attentions[0]
    for att in attentions[1:]:
        rollout = rollout * att

    return [rollout]

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
                    im = ax.imshow(attn[np.newaxis, :], aspect="auto", cmap="viridis")
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

def get_attention_maps(batch, model, device, rollout=False, average_heads=False, return_logits = False) -> tuple:
        
    with torch.no_grad():
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Forward pass with attention outputs, passing only batch["labels"]
        logits, attn_det, attn_time = model(
            batch["labels"],
            return_attn=True,
            average_heads=True if rollout else average_heads
        )

    # Detection transformer attention maps
    if rollout:
        attn_det = get_rollout_attention(attn_det)

    # Time transformer attention maps
    if rollout:
        attn_time = get_rollout_attention(attn_time)

    if return_logits:
        return logits, attn_det, attn_time
    return attn_det, attn_time

def compute_aggregated_metrics(dataloader, model, device, average_heads=False):
    """
    Computes aggregated metrics from the dataloader using the model.
    Prints aggregated metrics.
    """
    det_aggregated = {}
    time_aggregated = {}
    for batch in iter(dataloader):
        # Get attention maps
        attn_det, attn_time = get_attention_maps(batch, model, device, rollout=True, average_heads=average_heads)
        # Get max attention value for det and for time
        # For attn_det: shape (B, T, N), B=1, so attn_det[0] is (T, N)
        det_max_indices = torch.argmax(attn_det[0], dim=2)  # shape: (T,)
        #flatten batch (only 1)
        det_max_indices = det_max_indices[0]
        for idx in det_max_indices:
            idx = str(int(idx))
            if idx not in det_aggregated:
                det_aggregated[idx] = 0
            det_aggregated[idx] += 1

        max_idx = torch.argmax(attn_time[0][0])
        # Convert the flat index to two indices (i, j) for the (N, N) attention map
        i = max_idx // attn_time[0].shape[1]
        j = max_idx % attn_time[0].shape[1]
        for idx in [i, j]:
            idx = str(int(idx))
            if idx not in time_aggregated:
                time_aggregated[idx] = 0
            time_aggregated[idx] += 1
    
    print("Detection attention max values aggregated:")
    for k, v in sorted(det_aggregated.items(), key=lambda item: item[1], reverse=True):
        print(f"Max Det Attention: {k} - Count: {v}")

    print("\nTime attention max values aggregated:")
    for k, v in sorted(time_aggregated.items(), key=lambda item: item[1], reverse=True):
        print(f"Max Time Attention: {k} - Count: {v}")

def mega_slideshow(dataloader, model, device, top_k=5, threshold=0.025):
    """
    Collects all slides from the dataloader and shows a single mega slideshow.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF

    all_slides = []
    for batch in dataloader:
        boxes = batch['boxes'].to(device)
        images = batch['images'].to(device)
        logits, attn_det, _ = get_attention_maps(batch, model, device, rollout=True, return_logits=True)
        B, T, _, _, _ = images.shape
        _, _, N, _ = boxes.shape
        for b in range(B):
            for t in range(T):
                image_tensor = images[b, t]
                image = TF.to_pil_image(image_tensor.cpu())
                attention = attn_det[0][b, t]
                box_set = boxes[b, t]
                pred_t = torch.argmax(logits[b, t]).item()
                all_slides.append((image, attention, box_set, pred_t, b, t))

    num_slides = len(all_slides)
    colors = plt.cm.get_cmap('tab10', top_k)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.23, left=0.08, right=0.92)

    def draw_slide(idx):
        ax.clear()
        image, attention, box_set, pred_t, b, t = all_slides[idx]
        ax.imshow(image)
        ax.axis('off')
        topk = min(top_k, box_set.shape[0])
        topk_scores, topk_indices = torch.topk(attention, k=topk)
        for rank, (score, idx_box) in enumerate(zip(topk_scores, topk_indices)):
            if score.item() < threshold:
                continue
            box = box_set[idx_box].cpu().numpy()
            # box: (x1, y1, x2, y2) as absolute pixel coordinates in the original image
            x1, y1, x2, y2 = box
            img_w, img_h = image.size  # current (resized) image size
            # Assume original image size is MIN_SIZE x MAX_SIZE
            MIN_SIZE = 512
            MAX_SIZE = int(MIN_SIZE * 1.35)
            orig_w, orig_h = MAX_SIZE, MIN_SIZE
            # Scale coordinates from original to resized image
            scale_x = img_w / orig_w
            scale_y = img_h / orig_h
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            width = x2 - x1
            height = y2 - y1
            color = colors(rank)[:3]
            rect = patches.FancyBboxPatch(
                (x1, y1), width, height,
                boxstyle="round,pad=0.02",
                linewidth=3 + 2 * (topk - rank - 1),
                edgecolor=color,
                facecolor='none',
                alpha=0.85
            )
            ax.add_patch(rect)
            concept_idx = int(idx_box)
            concept_label = concept_names[concept_idx] if concept_idx < len(concept_names) else f"Box {concept_idx}"
            ax.text(
                x1, y1 - 8,
                f"{score.item():.3f}\n{concept_label}",
                color='black',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )
        ax.set_title(f"Batch {b}, Time {t} - Top {topk} Attended Boxes (score ≥ {threshold})\nPredicted class: {ego_actions_name[pred_t]}", fontsize=14)
        fig.canvas.draw_idle()

    ax_slide = plt.axes([0.18, 0.11, 0.64, 0.05])
    slider = Slider(ax_slide, 'Frame', 0, num_slides - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        draw_slide(idx)
    slider.on_changed(update)

    button_height = 0.06
    button_width = 0.13
    axprev = plt.axes([0.01, 0.01, button_width, button_height])
    axnext = plt.axes([0.86, 0.01, button_width, button_height])
    bnext = Button(axnext, 'Next', color='#e0e0e0', hovercolor='#b0b0b0')
    bprev = Button(axprev, 'Prev', color='#e0e0e0', hovercolor='#b0b0b0')

    for btn_ax in [axprev, axnext]:
        for spine in btn_ax.spines.values():
            spine.set_visible(False)
        btn_ax.set_facecolor('#f8f8f8')
        btn_ax.set_alpha(0.95)

    def next_slide(event):
        idx = int(slider.val)
        if idx < num_slides - 1:
            slider.set_val(idx + 1)
    def prev_slide(event):
        idx = int(slider.val)
        if idx > 0:
            slider.set_val(idx - 1)
    bnext.on_clicked(next_slide)
    bprev.on_clicked(prev_slide)

    draw_slide(0)
    plt.show()

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    rollout = True
    average_heads = False
    device = get_device()
    print("Working on device :", device)
    
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
    print("Starting to compute attention maps...")
    # Iterate over the dataloader and visualize for each batch
    mega_slideshow(dataloader, model, device, top_k=5, threshold=0.01)

if __name__ == "__main__":
    main()


