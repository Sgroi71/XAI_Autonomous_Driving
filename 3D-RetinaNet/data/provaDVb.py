import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import sys
sys.path.append('/home/fabio/dev/XAI_Autonomous_Driving/XbD/models')
from first_version import XbD_FirstVersion

# Assicurati che la classe videoDataset sia importata o definita nel file
# from your_module import videoDataset

# Esempio di implementazione minima della classe videoDataset se non esiste già
from dataset_prediction import VideoDataset
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
class Args:
    ANCHOR_TYPE = 'default'
    DATASET = 'road'  # o 'ucf24', 'ava'
    SUBSETS = ['val_3']
    SEQ_LEN = 8
    BATCH_SIZE = 1024
    MIN_SEQ_STEP = 1
    MAX_SEQ_STEP = 1
    DATA_ROOT = './dataset/'  # aggiorna con il tuo path
    PREDICTION_ROOT = './road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/detections-30-08-50'  # aggiorna con il tuo path
    MAX_ANCHOR_BOXES = 20
    NUM_CLASSES = 41

args = Args()
dataset = VideoDataset(args, train=True, input_type='rgb', transform=None, skip_step=1, full_test=False)


# Crea il DataLoader
dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)

inputs, labels = next(iter(dataloader))
B, S, A, C = inputs.shape  # BATCH_SIZE, SEQ_LEN, MAX_ANCHOR_BOXES, NUM_CLASSES
output_dim = 7  # N. EGO LABELS

# Define a model that processes per-anchor and reduces over anchors
class SimpleModel(nn.Module):
    def __init__(self, num_classes, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x: (B, S, A, C)
        x = self.fc1(x)         # (B, S, A, 32)
        x = self.relu(x)
        x = x.mean(dim=2)       # Reduce over anchors: (B, S, 32)
        x = self.fc2(x)         # (B, S, 7)
        return x

model = XbD_FirstVersion(20, args.MAX_ANCHOR_BOXES)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 500

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, output_dim), labels.view(-1))
        loss.backward()
        #print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

model.eval()
with torch.no_grad():
    # Select 10 random samples from the dataset
    indices = random.sample(range(len(dataset)), 10)
    for idx in indices:
        sample_inputs, sample_labels = dataset[idx]
        sample_inputs = sample_inputs.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(sample_inputs)
        preds = outputs.argmax(dim=-1)  # (1, S)
        print(f"Sample idx: {idx}")
        print(f"Predictions: {preds.cpu().numpy()}")
        print(f"Ground Truth: {sample_labels.numpy()}")
        print("-" * 40)

# Evaluate accuracy and additional metrics on the whole dataset

all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for inputs, labels in DataLoader(dataset, batch_size=args.BATCH_SIZE):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=-1)  # (B, S)
        mask = labels != -1  # Ignore padded/invalid labels
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())

if all_labels:

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Evaluation accuracy on whole dataset: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Weighted F1 score: {f1:.4f}")
    print("Classification report:")
    print(classification_report(all_labels, all_preds, digits=4))
else:
    print("No valid labels found for evaluation.")
