# inference/maskrcnn_runner.py
import torch
import numpy as np
import cv2
from object_detection.model import get_mask_rcnn_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Vehicle",
    "Damaged Building",
    "Damaged Road"
]

class MaskRCNNRunner:
    def __init__(self, weight_path, num_classes=len(CLASS_NAMES), backbone="resnet101"):
        self.model = get_mask_rcnn_model(num_classes=num_classes, backbone=backbone)
        checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(DEVICE).eval()
        print(f"[INFO] Mask R-CNN loaded from {weight_path}")

    @torch.no_grad()
    def run_on_pair(self, image, binary_mask, score_thresh=0.5, overlap_thresh=0.3):
        """
        Run Mask R-CNN inference on a single image + binary mask.
        Args:
            image (np.ndarray): (H,W,3) uint8 BGR image
            binary_mask (np.ndarray): (H,W) uint8 0/255 change mask
        Returns:
            dict with:
                - counts: {class: number of instances}
                - change_percent: {class: % of class area from total change area}
                - vis_image: visualization with detections
        """
        img_tensor = torch.from_numpy(image.transpose(2,0,1)).float().to(DEVICE) / 255.0
        outputs = self.model([img_tensor])[0]

        H, W = image.shape[:2]
        vis_img = image.copy()
        overlay = np.zeros_like(image, dtype=np.uint8)

        counts = {}
        change_area = np.sum(binary_mask > 0)
        class_change_area = {}

        for i in range(len(outputs["scores"])):
            score = outputs["scores"][i].item()
            if score < score_thresh:
                continue

            label_id = outputs["labels"][i].item()
            label = CLASS_NAMES[label_id]

            # predicted mask
            mask = outputs["masks"][i, 0].cpu().numpy() > 0.5
            # restrict to detected change
            restricted_mask = np.logical_and(mask, binary_mask > 0)

            if restricted_mask.sum() == 0:
                continue

            overlap_ratio = restricted_mask.sum() / (mask.sum() + 1e-6)
            if overlap_ratio < overlap_thresh:
                continue

            # count objects
            counts[label] = counts.get(label, 0) + 1
            # accumulate changed pixels by class
            class_change_area[label] = class_change_area.get(label, 0) + restricted_mask.sum()

            # visualization
            color = np.random.randint(0, 255, size=3).tolist()
            overlay[restricted_mask] = color

            x1, y1, x2, y2 = outputs["boxes"][i].cpu().numpy().astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{label} {score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        vis_img = cv2.addWeighted(vis_img, 1.0, overlay, 0.5, 0)

        # calculate per-class percentage relative to total change area
        change_percent = {}
        for lbl, area in class_change_area.items():
            change_percent[lbl] = (area / max(1, change_area)) * 100

        return {
            "counts": counts,
            "change_percent": change_percent,
            "vis_image": vis_img
        }

    @torch.no_grad()
    def run_on_tiles(self, img_tiles, mask_tiles, score_thresh=0.5, overlap_thresh=0.3):
        """
        Run Mask R-CNN inference on multiple tiles.
        Returns a list of results (dicts).
        """
        results = []
        for img, mask in zip(img_tiles, mask_tiles):
            res = self.run_on_pair(img, mask, score_thresh, overlap_thresh)
            results.append(res)
        return results
