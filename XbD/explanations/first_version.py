import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/home/jovyan/python/XAI_Autonomous_Driving/")
from XbD.models.first_version import XbD_FirstVersion
from XbD.data.dataset_prediction import VideoDataset



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


def main():
    N = 10  # number of objects per time step
    batch_size = 1
    device = get_device()

    concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']
    ego_actions_name = ['AV-Stop', 'AV-Mov', 'AV-TurRht', 'AV-TurLft', 'AV-MovRht', 'AV-MovLft', 'AV-Ovtak']

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

    dataset_val = VideoDataset(args, train=False, skip_step=args.SEQ_LEN)
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

            # now iterate detections as before
            for det_expl in time_expls['detections']:
                det_id = det_expl['detection']
                print(f"  Detection {det_id}:")

                agent_name, agent_val, agent_conf = det_expl['agent']
                print(f"    Agent: {agent_name} (contrib {agent_val:+.2f}, conf {agent_conf:.2f})")

                print("    Actions:")
                for name, val, conf in det_expl['actions']:
                    print(f"      {name}: contrib {val:+.2f}, conf {conf:.2f}")

                print("    Locations:")
                for name, val, conf in det_expl['locations']:
                    print(f"      {name}: contrib {val:+.2f}, conf {conf:.2f}")

if __name__ == "__main__":
    main()


