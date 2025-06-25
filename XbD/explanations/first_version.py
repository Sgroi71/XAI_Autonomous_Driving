import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/jovyan/python/XAI_Autonomous_Driving/")
from XbD.models.first_version import XbD_FirstVersion
from XbD.data.dataset_prediction import VideoDataset
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
import numpy as np


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



def save_attention_images(batch, device, output_dir, attended_ids):

    os.makedirs(output_dir, exist_ok=True)

    boxes = batch['boxes'].to(device)
    images = batch['images'].to(device)
    B, T, _, _, _ = images.shape
    _, _, N, _ = boxes.shape
    for b in range(B):
        for t in range(T):
            image_tensor = images[b, t]
            image = TF.to_pil_image(image_tensor.cpu())
            box_set = boxes[b, t]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            ax.axis('off')
            for rank,idx_box in enumerate(attended_ids):
                box = box_set[idx_box].cpu().numpy()
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
                color = plt.cm.tab10(rank % 10)[:3]
                rect = patches.FancyBboxPatch(
                    (x1, y1), width, height,
                    boxstyle="round,pad=0.02",
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.85
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 4,
                    f"{idx_box} ({rank+1})",
                    color='black',
                    fontsize=7,
                    fontweight='normal',
                    bbox=dict(facecolor=color, alpha=0.25, edgecolor='none', boxstyle='round,pad=0.1')
                )
            ax.set_title(f"Batch {b}, Time {t} - Top {len(attended_ids)} Attended Boxes", fontsize=9)
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"attn_b{b}_t{t}.png")
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
# Esempio di utilizzo nel main:
# save_attention_images(dataloader, model, device, output_dir="output_attn_images", top_k_det=10, det_attn_treshold=0, max_images=100)


def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()

    concept_names = [
    'Pedestrian', 'Car', 'Cyclist', 'Motorbike', 'MedicalVehicle', 'LargeVehicle', 'Bus', 'EmergencyVehicle',
    'TrafficLight', 'OtherTrafficLight', 'RedLight', 'AmberLight', 'GreenLight',
    'MovingAway', 'MovingTowards', 'Moving', 'Braking', 'Stopped',
    'LeftIndicator', 'RightIndicator', 'HazardLights', 'TurningLeft', 'TurningRight', 'Overtaking',
    'WaitingToCross', 'CrossingFromLeft', 'CrossingFromRight', 'Crossing', 'PushingObject',
    'VehicleLane', 'OutgoingLane', 'OutgoingCycleLane', 'IncomingLane', 'IncomingCycleLane',
    'Pavement', 'LeftPavement', 'RightPavement', 'Junction', 'PedestrianCrossing', 'BusStop', 'ParkingArea'
    ]

    ego_actions_name = [
        'Stop', 'Moving', 'TurnRight', 'TurnLeft',
        'MoveRight', 'MoveLeft', 'Overtake'
    ]
    model = load_model_weights(
        XbD_FirstVersion,
        f"{ROOT}XbD/results/version1/best_model_weights.pth",
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

    explanations = model.explain_contributions(
        labels=batch["labels"].to(device),
        concept_names=concept_names,
        ego_action_names=ego_actions_name,
        top_k_dets=5,
        top_k_actions=1,
        top_k_locations=1
    )



    for b_idx, batch_expls in enumerate(explanations):
        print(f"Batch {b_idx}:")
        for t_idx, time_expls in enumerate(batch_expls):
            # pull out the ego‐maneuver prediction
            pred_ego = time_expls['predicted_ego']
            print(f" Time step {t_idx}: Predicted ego → {pred_ego}")
            attended_ids=[]
            # now iterate detections as before
            for det_expl in time_expls['detections']:
                det_id = det_expl['detection']
                print(f"  Detection {det_id}:")
                if det_id not in attended_ids:
                    attended_ids.append(det_id)
                else:
                    continue

                agent_name, agent_val, agent_conf = det_expl['agent']
                print(f"    Agent: {agent_name} (contrib {agent_val:+.2f}, conf {agent_conf:.2f})")

                print("    Actions:")
                for name, val, conf in det_expl['actions']:
                    print(f"      {name}: contrib {val:+.2f}, conf {conf:.2f}")

                print("    Locations:")
                for name, val, conf in det_expl['locations']:
                    print(f"      {name}: contrib {val:+.2f}, conf {conf:.2f}")
    save_attention_images(batch, device, output_dir="{ROOT}/XbD/results_F1/version1/explanation/", attended_ids=attended_ids)


if __name__ == "__main__":
    main()


