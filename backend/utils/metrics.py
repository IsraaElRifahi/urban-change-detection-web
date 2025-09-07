# utils/metrics.py
import torch
import numpy as np

def compute_metrics(preds, labels):
    """
    Compute IoU and F1 Score between predictions and ground truth.
    Args:
        preds (list of torch.Tensor): List of predicted masks (B, H, W)
        labels (list of torch.Tensor): List of ground truth masks (B, H, W)
    Returns:
        dict: IoU and F1 score as floats
    """
    preds = torch.cat(preds).cpu().numpy().astype(np.uint8)
    labels = torch.cat(labels).cpu().numpy().astype(np.uint8)

    tp = np.logical_and(preds == 1, labels == 1).sum()
    fp = np.logical_and(preds == 1, labels == 0).sum()
    fn = np.logical_and(preds == 0, labels == 1).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

    return {
        'IoU': round(iou, 4),
        'F1': round(f1, 4)
    }