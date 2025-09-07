# model.py
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_mask_rcnn_model(num_classes, backbone="resnet101", pretrained=True):
    """
    Returns a Mask R-CNN model with specified backbone and number of classes.
    
    Args:
        num_classes (int): Number of classes (including background)
        backbone (str): 'resnet50' or 'resnet101'
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        MaskRCNN model
    """
    if backbone == 'resnet50':
        # Use torchvision helper for resnet50
        model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        # Replace classifier for our num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = \
            torch.nn.Sequential()  # placeholder, will replace next
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # Mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    elif backbone == 'resnet101':
        # Build custom backbone
        backbone_model = resnet_fpn_backbone('resnet101', pretrained=pretrained)
        model = MaskRCNN(backbone_model, num_classes=num_classes)
    else:
        raise ValueError("Backbone must be 'resnet50' or 'resnet101'")
    
    return model
