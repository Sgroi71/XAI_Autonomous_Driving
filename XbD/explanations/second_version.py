import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/fabio/XAI_Autonomous_Driving/")
from XbD.models.second_version import XbD_SecondVersion
from XbD.data.dataset_prediction_bounded_test import VideoDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import json
from matplotlib.widgets import Slider, Button
import numpy as np

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
def retrive_best_configuration(filename):
    
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

def get_attention_maps(batch, model, device, rollout=False, average_heads=False, return_logits = False) -> tuple:
        
    with torch.no_grad():
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Forward pass with attention outputs, passing only batch["labels"]
        logits, attn_det = model(
            batch["labels"],
            return_attn=True,
            average_heads=True if rollout else average_heads
        )

    # Detection transformer attention maps
    if rollout:
        attn_det = get_rollout_attention(attn_det)


    if return_logits:
        return logits, attn_det
    return attn_det

def mega_slideshow(dataloader, model, device, top_k_det=5, det_attn_treshold=0.025, top_k_frames=3, slide_limit = 400):
    """
    Collects all slides from the dataloader and shows a single mega slideshow.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF

    all_slides = []
    for slide_num, batch in enumerate(dataloader):
        labels = batch["labels"]
        boxes = batch['boxes'].to(device)
        images = batch['images'].to(device)
        logits, attn_det = get_attention_maps(batch, model, device, rollout=True, return_logits=True)
        assert len(attn_det) == 1, "Expected only one layer of attention maps after rollout"
        attn_det = attn_det[0]
        B, T, _, _, _ = images.shape
        _, _, N, _ = boxes.shape
        for b in range(B):
            for t in range(T):
                image_tensor = images[b, t]
                image = TF.to_pil_image(image_tensor.cpu())
                # Here we take only the first layer, cause rollout give only a layer
                
                attention_det = attn_det[b, t]
                box_set = boxes[b, t]
                label_set = labels[b, t]
                pred_t = torch.argmax(logits[b, t]).item()
                all_slides.append((image, attention_det, box_set, label_set, pred_t, b, t))
        if slide_num + 1 >= slide_limit:
            break

    num_slides = len(all_slides)
    colors = plt.cm.get_cmap('tab10', top_k_det)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.23, left=0.08, right=0.92)

    def draw_slide(idx):
        ax.clear()
        image, attention_det, box_set, label_set, pred_t, b, t = all_slides[idx]
        ax.imshow(image)
        ax.axis('off')
        topk = min(top_k_det, box_set.shape[0])
        topk_scores, topk_indices = torch.topk(attention_det, k=topk)
        for rank, (score, idx_box) in enumerate(zip(topk_scores, topk_indices)):
            if score.item() < det_attn_treshold:
                continue
            box = box_set[idx_box].cpu().numpy()
            label = label_set[idx_box]
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
                linewidth=1,
                edgecolor=color,
                facecolor='none',
                alpha=0.85
            )
            ax.add_patch(rect)
            # label: vector of 41 elements
            label_vec = label.cpu().numpy()
            agent_idx = np.argmax(label_vec[:10])
            action_idx = np.argmax(label_vec[10:29]) + 10
            location_idx = np.argmax(label_vec[29:]) + 29
            concept_label = f"{concept_names[agent_idx]}-{concept_names[action_idx]}-{concept_names[location_idx]}"
            ax.text(
                x1, y1 - 4,
                f"{score.item():.2f}\n{concept_label}",
                color='black',
                fontsize=7,
                fontweight='normal',
                bbox=dict(facecolor=color, alpha=0.25, edgecolor='none', boxstyle='round,pad=0.1')
            )

        ax.set_title(f"Batch {b}, Time {t} - Top {topk} Attended Boxes (score ≥ {det_attn_treshold})", fontsize=9)
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
    device = get_device()

    cof_id, params = retrive_best_configuration(f"{ROOT}XbD/results_F1/version2/grid_results.json")
    model = load_model_weights(
        XbD_SecondVersion,
        f"{ROOT}XbD/results_F1/version2/grid_{cof_id}/best_model_v2_weights.pth",
        get_device(),
        num_classes=41,
        d_model=params.get("d_model", 64),
        nhead=params.get("nhead", 2),
        num_layers=params.get("num_layers", 1),
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

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN,explaination=True)
    dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    print("Created dataloader with {} samples".format(len(dataloader)))
    print("Starting to compute attention maps...")
    # Iterate over the dataloader and visualize for each batch
    mega_slideshow(dataloader, model, device, top_k_det=10, det_attn_treshold=0, slide_limit=100)

if __name__ == "__main__":
    main()


