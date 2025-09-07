import torch.nn as nn

def CrossEntropyLoss():
    """
    Returns a standard CrossEntropyLoss instance.

    Suitable for pixel-wise binary or multi-class classification.
    This loss expects raw logits as input and applies LogSoftmax + NLL internally.
    """
    return nn.CrossEntropyLoss()