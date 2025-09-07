import torch
import cv2
import numpy as np
from stransunt.stransunet import STransUNet
from utils.postprocessing import extract_changed_regions, calculate_change_percentage
from utils.crf import apply_crf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STransUNetRunner:
    def __init__(self, model_path: str):
        """Load STransUNet model from checkpoint."""
        self.model = STransUNet().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"[INFO] Loaded STransUNet model from {model_path}")

    @torch.no_grad()
    def run_on_pair(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """
        Run change detection on a single image pair (tile).
        
        Args:
            img1: Before image (H,W,3), uint8
            img2: After image (H,W,3), uint8
        
        Returns:
            binary_mask (np.ndarray): 2D mask (0/1)
            changed_rgb (np.ndarray): RGB image showing only changed areas
            change_percent (float): % of changed pixels
        """
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Convert images to tensor
        img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        img1_tensor = (img1_tensor - 0.5) / 0.5  # normalize
        img2_tensor = (img2_tensor - 0.5) / 0.5

        img1_tensor = img1_tensor.to(DEVICE)
        img2_tensor = img2_tensor.to(DEVICE)

        # Run model
        outputs = self.model(img1_tensor, img2_tensor)        # (1, 2, H, W)
        probs = torch.softmax(outputs, dim=1)                 # softmax
        change_prob = probs[0, 1, :, :].cpu().numpy()         # class=1 prob map

        # ---- Apply CRF ----
        h, w = change_prob.shape
        rgb_for_crf = cv2.resize(img2, (w, h))  # ensure match
        refined_mask = apply_crf(
        rgb_for_crf,
        change_prob,
        sxy_gaussian=3,
        compat_gaussian=3,
        sxy_bilateral=20,
        srgb_bilateral=12,
        compat_bilateral=2,
        iterations=2
)

        # Convert to 0/255 for visualization
        binary_mask = refined_mask.astype(np.uint8)           # 0/1
        binary_mask_visual = binary_mask * 255

        # Extract RGB regions and calculate % change
        changed_rgb = extract_changed_regions(binary_mask_visual, img2)
        change_percent = calculate_change_percentage(binary_mask)

        return binary_mask, changed_rgb, change_percent

    @torch.no_grad()
    def run_on_tiles(self, img1_tiles: list, img2_tiles: list) -> tuple:
        """
        Run model on multiple tiles and return lists of results.
        Args:
            img1_tiles: list of np.ndarray before images
            img2_tiles: list of np.ndarray after images
        Returns:
            binary_masks: list of 2D masks
            changed_rgbs: list of RGB images showing changed areas
            change_percents: list of % change per tile
        """
        binary_masks = []
        changed_rgbs = []
        change_percents = []

        for t1, t2 in zip(img1_tiles, img2_tiles):
            b_mask, c_rgb, c_pct = self.run_on_pair(t1, t2)
            binary_masks.append(b_mask)
            changed_rgbs.append(c_rgb)
            change_percents.append(c_pct)

        return binary_masks, changed_rgbs, change_percents
