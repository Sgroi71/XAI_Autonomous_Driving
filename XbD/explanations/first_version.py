import torch
from torch.utils.data import DataLoader

from XbD.models.first_version import XbD_FirstVersion
from XbD.data.fake_dataset import FakeDataset

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
    N = 5  # number of objects per time step
    batch_size = 1
    device = get_device()

    concept_names = ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL', 'Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj', 'VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking']

    model = load_model_weights(
        XbD_FirstVersion,
        "/Users/marcodonnarumma/Desktop/XAI_Autonomous_Driving/XbD/xbdfirstversion_weights.pth",
        get_device(),
        num_classes=41,
        N=N
    )

    samples = FakeDataset(length=2000, T=8, N=N)  # Example dataset with 2 samples, 8 time steps, N objects

    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,       # adjust num_workers as needed
        pin_memory=True if device.type == "cuda" else False
    )

    batch = next(iter(dataloader))

    explanations = model.explain_contributions(
        labels=batch["labels"],
        concept_names=concept_names)

    for b_idx, batch_expls in enumerate(explanations):
        print(f"Batch {b_idx}:")
        for t_idx, time_expls in enumerate(batch_expls):
            print(f" Time step {t_idx}:")
            for det_expl in time_expls:
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


