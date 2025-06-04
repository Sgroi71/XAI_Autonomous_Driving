import torch
from torch.utils.data import DataLoader

# Assicurati che la classe videoDataset sia importata o definita nel file
# from your_module import videoDataset

# Esempio di implementazione minima della classe videoDataset se non esiste già
from dataset_prediction import VideoDataset
class Args:
    ANCHOR_TYPE = 'default'
    DATASET = 'road'  # o 'ucf24', 'ava'
    SUBSETS = ['train','val']
    SEQ_LEN = 8
    BATCH_SIZE = 2
    MIN_SEQ_STEP = 1
    MAX_SEQ_STEP = 1
    DATA_ROOT = '/home/jovyan/nfs/lsgroi/dataset/'  # aggiorna con il tuo path
    ANNO_ROOT = '/home/jovyan/nfs/lsgroi/dataset/'  # aggiorna con il tuo path
    PREDICTION_ROOT = '/home/jovyan/python/XAI_Autonomous_Driving/3D-RetinaNet/road'  # aggiorna con il tuo path

args = Args()
dataset = VideoDataset(args, train=True, input_type='rgb', transform=None, skip_step=1, full_test=False)


# Crea il DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Stampa i risultati
for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print("Inputs shape:", inputs.shape)
    print("Labels:", labels)