import torch.nn.functional as F
import torch

# taken from 3D-RetinaNet/modules/detection_loss.py
# loss used for ego_labels

def sigmoid_focal_loss(preds, labels, num_pos, alpha, gamma):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''
    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = preds * labels + (1.0 - preds) * (1.0 - labels)
    focal_weight = alpha_factor * ((1-pt) ** gamma)
    loss = (loss * focal_weight).sum() / num_pos
    return loss


def get_one_hot_labels(tgt_labels, numc):
    new_labels = torch.zeros([tgt_labels.shape[0], numc], device=tgt_labels.device)
    new_labels[:, tgt_labels] = 1.0
    return new_labels

def ego_loss(ego_preds, ego_labels, alpha=0.25, gamma=2.0):
    mask = ego_labels > -1 # For our models (XbD) they should all be > -1, but this is kept to be faithful to the original code
    numc = ego_preds.shape[-1]
    masked_preds = ego_preds[mask].reshape(-1, numc)  # Remove Ignore preds
    masked_labels = ego_labels[mask].reshape(-1)  # Remove Ignore labels
    one_hot_labels = get_one_hot_labels(masked_labels, numc)
    ego_loss = 0
    if one_hot_labels.shape[0] > 0:
        ego_loss = sigmoid_focal_loss(masked_preds, one_hot_labels, one_hot_labels.shape[0], alpha, gamma)

    return ego_loss


