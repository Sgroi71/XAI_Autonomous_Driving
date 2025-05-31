import sys
import pickle
import os
import numpy as np

print(f"Current working directory: {os.getcwd()}")
if len(sys.argv) != 2:
    print("Usage: python visualize_pred.py <input_file.pkl>")
    sys.exit(1)

input_file = sys.argv[1]
print(f"Input pkl file: {input_file}")

with open(input_file, "rb") as f:
    data = pickle.load(f)

""" 
I primi 4 valori di main sono le bounding boxes, i restanti 149 sono i logit delle classi.

Primo coord x del corner in alto a sinistra
Secondo coord y del corner in alto a sinistra
Coordinata x del corner in basso a destra
Coordinata y del corner in basso a destra

"""
import matplotlib.pyplot as plt
all_classes = [['agent_ness'], ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL'], ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj'], ['VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking'], ['Bus-MovAway', 'Bus-MovTow', 'Bus-Stop', 'Bus-XingFmLft', 'Car-Brake', 'Car-IncatLft', 'Car-IncatRht', 'Car-MovAway', 'Car-MovTow', 'Car-Stop', 'Car-TurLft', 'Car-TurRht', 'Car-XingFmLft', 'Car-XingFmRht', 'Cyc-MovAway', 'Cyc-MovTow', 'Cyc-Stop', 'Cyc-TurLft', 'Cyc-XingFmLft', 'Cyc-XingFmRht', 'LarVeh-Stop', 'MedVeh-IncatLft', 'MedVeh-MovTow', 'MedVeh-Stop', 'MedVeh-TurRht', 'OthTL-Green', 'OthTL-Red', 'Ped-Mov', 'Ped-MovAway', 'Ped-MovTow', 'Ped-PushObj', 'Ped-Stop', 'Ped-Wait2X', 'Ped-Xing', 'Ped-XingFmLft', 'Ped-XingFmRht', 'TL-Amber', 'TL-Green', 'TL-Red'], ['Bus-MovTow-IncomLane', 'Bus-MovTow-Jun', 'Bus-Stop-IncomLane', 'Bus-Stop-VehLane', 'Bus-XingFmLft-Jun', 'Car-Brake-Jun', 'Car-Brake-VehLane', 'Car-IncatLft-Jun', 'Car-IncatLft-VehLane', 'Car-IncatRht-IncomLane', 'Car-IncatRht-Jun', 'Car-MovAway-Jun', 'Car-MovAway-OutgoLane', 'Car-MovAway-VehLane', 'Car-MovTow-IncomLane', 'Car-MovTow-Jun', 'Car-Stop-IncomLane', 'Car-Stop-Jun', 'Car-Stop-VehLane', 'Car-TurLft-Jun', 'Car-TurLft-VehLane', 'Car-TurRht-IncomLane', 'Car-TurRht-Jun', 'Car-XingFmLft-Jun', 'Cyc-MovAway-Jun', 'Cyc-MovAway-LftPav', 'Cyc-MovAway-OutgoCycLane', 'Cyc-MovAway-OutgoLane', 'Cyc-MovAway-VehLane', 'Cyc-MovTow-IncomCycLane', 'Cyc-MovTow-IncomLane', 'Cyc-MovTow-Jun', 'Cyc-MovTow-LftPav', 'Cyc-Stop-IncomCycLane', 'Cyc-Stop-IncomLane', 'Cyc-Stop-Jun', 'Cyc-TurLft-Jun', 'Cyc-XingFmLft-Jun', 'MedVeh-MovTow-IncomLane', 'MedVeh-MovTow-Jun', 'MedVeh-Stop-IncomLane', 'MedVeh-Stop-Jun', 'MedVeh-TurRht-Jun', 'Ped-Mov-Pav', 'Ped-MovAway-LftPav', 'Ped-MovAway-Pav', 'Ped-MovAway-RhtPav', 'Ped-MovTow-IncomLane', 'Ped-MovTow-LftPav', 'Ped-MovTow-RhtPav', 'Ped-MovTow-VehLane', 'Ped-PushObj-LftPav', 'Ped-PushObj-RhtPav', 'Ped-Stop-BusStop', 'Ped-Stop-LftPav', 'Ped-Stop-Pav', 'Ped-Stop-RhtPav', 'Ped-Stop-VehLane', 'Ped-Wait2X-LftPav', 'Ped-Wait2X-RhtPav', 'Ped-XingFmLft-IncomLane', 'Ped-XingFmLft-Jun', 'Ped-XingFmLft-VehLane', 'Ped-XingFmLft-xing', 'Ped-XingFmRht-IncomLane', 'Ped-XingFmRht-Jun', 'Ped-XingFmRht-RhtPav', 'Ped-XingFmRht-VehLane']]

for key, value in data.items():
    shape = getattr(value, 'shape', None)
    print(f"Key: {key}, Shape: {shape}")
    if key == 'main':
        print("MAIN")
        # Remove bounding boxes
        concepts = value[:, 4:]
        assert concepts.shape[1] == 149
        # Associamo ogni logit alla sua corrispondente classe in all_classes
        flattened_classes = [cls for group in all_classes for cls in group]
        logits_by_class = {cls: concepts[:, i] for i, cls in enumerate(flattened_classes)}
        threshold = 0.5
        for cls, logits in logits_by_class.items():
            for idx, logit in enumerate(logits):
                if logit > threshold:
                    print(f"Class: {cls}, Anchor box: {idx}, Logit: {logit}")
        

            
