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
from matplotlib.widgets import Slider, Button
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import gc
import torch


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
        model.eval()
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Forward pass with attention outputs, passing only batch["labels"]
        logits, attn_det, attn_time = model(
            batch["labels"],
            return_attn=True,
            average_heads= True if rollout else average_heads
        )

    # Detection transformer attention maps
    if rollout:
        attn_det = get_rollout_attention(attn_det)

    # Time transformer attention maps
    if rollout:
        attn_time = get_rollout_attention(attn_time)

    # If rollout is False and average_head = True, take the last layer attention maps
    if not rollout and average_heads:
        attn_det = [attn_det[0]]
        attn_time = [attn_time[0]]

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

def mega_slideshow(dataloader, model, device, top_k_det=5, det_attn_treshold=0, top_k_frames=3, max_clips_per_slideshow=400, rollout = True, average_heads=True):
    """
    Collects slides from the dataloader and shows them in multiple slideshows,
    each with up to max_clips_per_slideshow slides, until the dataloader is exhausted.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torchvision.transforms.functional as TF

    def show_slideshow(slides, slideshow_idx=0):
        num_slides = len(slides)
        colors = plt.cm.get_cmap('tab10', top_k_det)
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.23, left=0.08, right=0.92)

        def draw_slide(idx):
            ax.clear()
            image, attention_det, attention_frames, box_set, label_set, pred_t, gt, b, t = slides[idx]
            ax.imshow(image)
            ax.axis('off')
            # Apply softmax to attention_det scores before topk
            if attention_det.numel() > 0:
                attention_det = attention_det - attention_det.min()
                if attention_det.max() > 0:
                    attention_det = attention_det / attention_det.max()
                attention_det = attention_det
            #apply softmax
            attention_det = torch.softmax(attention_det, dim=0)

            topk = min(top_k_det, box_set.shape[0])
            topk_scores, topk_indices = torch.topk(attention_det, k=topk)
            for rank, (score, idx_box) in enumerate(zip(topk_scores, topk_indices)):
                if score.item() < det_attn_treshold:
                    continue
                box = box_set[idx_box].cpu().numpy()
                label = label_set[idx_box]
                x1, y1, x2, y2 = box
                img_w, img_h = image.size
                MIN_SIZE = 512
                MAX_SIZE = int(MIN_SIZE * 1.35)
                orig_w, orig_h = MAX_SIZE, MIN_SIZE
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
                label_vec = label.cpu().numpy()
                agent_idx = np.argmax(label_vec[:10])
                action_idx = np.argmax(label_vec[10:29]) + 10
                location_idx = np.argmax(label_vec[29:]) + 29
                concept_label = f"{concept_names[agent_idx]}-{concept_names[action_idx]}-{concept_names[location_idx]}"
                # Offset label vertically to avoid overlap with the box
                label_y = y1 - 8 if y1 - 18 > 0 else y2 + 4
                # Display the label text vertically (rotated 90 degrees)
                # Draw the patch but make it invisible initially
                rect.set_visible(False)
                ax.add_patch(rect)
                # Always show the box
                rect.set_visible(True)
                ax.add_patch(rect)

                # Always show the score (outside the box, top-left)
                score_text = ax.text(
                    x1, y1 - 8 if y1 - 18 > 0 else y2 + 4,
                    f"{(score.item()*100):.1f}%",
                    color='black',
                    fontsize=7,
                    fontweight='bold',
                    rotation=0,
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, boxstyle='round,pad=0.05'),
                    visible=True
                )

                # Concept label (hidden by default, shown on hover or click)
                concept_text = ax.text(
                    x1 + width / 2, y1 - 30 if y1 - 40 > 0 else y2 + 16,
                    concept_label,
                    color='black',
                    fontsize=7,
                    fontweight='normal',
                    rotation=0,
                    va='bottom',
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, boxstyle='round,pad=0.05'),
                    visible=False
                )

                # Store for hover/click interactivity
                if not hasattr(ax, "_hover_patches"):
                    ax._hover_patches = []
                ax._hover_patches.append({
                    "rect": rect,
                    "concept_text": concept_text,
                    "x1": x1,
                    "y1": y1,
                    "width": width,
                    "height": height,
                })

            # Add hover/click interactivity after all patches are created
            def on_move(event):
                if not hasattr(ax, "_hover_patches"):
                    return
                for patch in ax._hover_patches:
                    patch["concept_text"].set_visible(False)
                    if event.inaxes == ax and patch["x1"] <= event.xdata <= patch["x1"] + patch["width"] and patch["y1"] <= event.ydata <= patch["y1"] + patch["height"]:
                        patch["concept_text"].set_visible(True)
                fig.canvas.draw_idle()

            def on_click(event):
                if not hasattr(ax, "_hover_patches"):
                    return
                # Keep concept_texts on click anywhere
                # Remove the patch whose box was clicked from ax._hover_patches
                for patch in list(ax._hover_patches):
                    if event.inaxes == ax and patch["x1"] <= event.xdata <= patch["x1"] + patch["width"] and patch["y1"] <= event.ydata <= patch["y1"] + patch["height"]:
                        patch["concept_text"].set_visible(True)
                        ax._hover_patches.remove(patch)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_move)
            fig.canvas.mpl_connect('button_press_event', on_click)

            # Normalize attention_frames before softmax
            if attention_frames.numel() > 0:
                attention_frames = attention_frames - attention_frames.min()
                if attention_frames.max() > 0:
                    attention_frames = attention_frames / attention_frames.max()
            # Normalize attention_frames scores before topk
            if attention_frames.numel() > 0:
                attention_frames = torch.softmax(attention_frames, dim=0)

            show_frames = min(top_k_frames, attention_frames.shape[0])
            topk_frame_scores, topk_frame_indices = torch.topk(attention_frames, k=show_frames)
            # Shift indices by +1 if index >= t-1
            topk_frame_indices = topk_frame_indices.cpu().numpy()
            topk_frame_indices = np.array([idx + 1 if idx >= t else idx for idx in topk_frame_indices])
            topk_frame_scores = topk_frame_scores.cpu().numpy()
            topk_frame_labels = [f"F{idx}" for idx in topk_frame_indices]
            topk_frame_info = ", ".join([f"{label} ({score:.2f})" for label, score in zip(topk_frame_labels, topk_frame_scores)])

            ax.set_title(
                f"Batch {b}, Time {t} - Top {topk} Attended Boxes (score ≥ {det_attn_treshold})\n"
                f"Predicted class: {ego_actions_name[pred_t]} GT: {ego_actions_name[gt]}\n"
                f"Most attended frames: {topk_frame_info}\n"
                f"Slideshow {slideshow_idx+1}, Slide {idx+1}/{num_slides}",
                fontsize=9
            )
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

    # Main loop: collect slides and show in batches
    all_slides = []
    all_gt = []
    all_pred = []
    slideshow_idx = 0
    for batch in dataloader:
        labels = batch["labels"]
        boxes = batch['boxes'].to(device)
        images = batch['images'].to(device)
        ego_label_gt = batch['ego_labels']
        logits, attn_det, frame_attention = get_attention_maps(batch, model, device, rollout=rollout, average_heads=average_heads, return_logits=True)
        assert len(attn_det) == 1, "Expected only one layer of attention maps after rollout"
        assert len(frame_attention) == 1, "Expected only one layer of time attention maps after rollout"
        attn_det = attn_det[0]
        frame_attention = frame_attention[0]
        B, T, _, _, _ = images.shape
        _, _, N, _ = boxes.shape
        for b in range(B):
            for t in range(T):
                image_tensor = images[b, t]
                image = TF.to_pil_image(image_tensor.cpu())
                attention_det = attn_det[b, t]
                attention_frames = torch.cat([frame_attention[b, t, :t], frame_attention[b, t, t+1:]], dim=0)
                box_set = boxes[b, t]
                label_set = labels[b, t]
                pred_t = torch.argmax(logits[b, t]).item()
                gt_t = int(ego_label_gt[b, t].cpu())
                if gt_t != -1:
                    all_pred.append(pred_t)
                    all_gt.append(gt_t)
                    all_slides.append((image, attention_det, attention_frames, box_set, label_set, pred_t, ego_label_gt[b, t], b, t))
        # Show slideshow if enough slides collected
        while len(all_slides) >= max_clips_per_slideshow:
            slides_to_show = all_slides[:max_clips_per_slideshow]
            # Per-slideshow classification report
            slideshow_gt = all_gt[slideshow_idx*max_clips_per_slideshow:]
            slideshow_pred = all_pred[slideshow_idx*max_clips_per_slideshow:]
            print("Classification report for slideshow:")
            print(classification_report(
            slideshow_gt, slideshow_pred,
            labels=list(range(len(ego_actions_name))),
            target_names=ego_actions_name,
            zero_division=0
            ))
            mf1 = f1_score(slideshow_gt, slideshow_pred, average='macro', zero_division=0)
            print(f"Macro F1 score for slideshow: {mf1:.4f}")
            #show_slideshow(slides_to_show, slideshow_idx)
            slideshow_idx += 1
            all_slides = all_slides[max_clips_per_slideshow:]
            # Free unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    # Show any remaining slides
    if all_slides:
        show_slideshow(all_slides, slideshow_idx)

    print("Final classification report:")
    print(classification_report(
        all_gt, all_pred,
        labels=list(range(len(ego_actions_name))),
        target_names=ego_actions_name,
        zero_division=0
    ))
    mf1 = f1_score(all_gt, all_pred, average='macro', zero_division=0)
    print(f"Macro F1 score for slideshow: {mf1:.4f}")

def main():
    N = 10  # number of objects per time step
    batch_size = 1
    rollout = True
    average_heads = True
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
    mega_slideshow(dataloader, model, device, top_k_det=10, det_attn_treshold=0, top_k_frames=3, max_clips_per_slideshow=200, rollout=rollout, average_heads=average_heads)

if __name__ == "__main__":
    main()


