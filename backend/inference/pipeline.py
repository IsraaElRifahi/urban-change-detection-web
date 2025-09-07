# pipeline.py
import cv2
import os
import numpy as np
from utils.image_splitter import split_image_to_tiles, merge_tiles_to_image_from_list, merge_tiles_to_image_from_folder
from utils.file_handler import cleanup_folder
from inference.runner_stransunet import STransUNetRunner
from inference.maskrcnn_runner import MaskRCNNRunner   # ✅ integrate maskrcnn

TILE_FOLDER = "temp_tiles"


class ChangeDetectionPipeline:
    def __init__(self, stransunet_model_path: str, maskrcnn_weight_path: str):
        self.stransunet = STransUNetRunner(stransunet_model_path)
        self.maskrcnn = MaskRCNNRunner(maskrcnn_weight_path)   # load Mask R-CNN

    def run(self, before_img_path: str, after_img_path: str):
        """
        Run the full pipeline:
        1. Pixel-level change detection (STransUNet).
        2. Object-level classification (Mask R-CNN).
        
        Returns:
            final_binary_mask: merged mask (0/255)
            final_changed_rgb: merged changed RGB image
            total_change_percent: % change for the whole image
            class_counts: dict {class_name: count}
            class_percentages: dict {class_name: % of total change}
            vis_img: visualization of Mask R-CNN detections
        """
        # Clear temp folder
        cleanup_folder(TILE_FOLDER)

        # Load images
        img1 = cv2.imread(before_img_path)
        img2 = cv2.imread(after_img_path)
        if img1 is None or img2 is None:
            raise ValueError("❌ Cannot read input images")

        original_size = img1.shape[:2]  # (H, W)

        # ----------------------------------
        # Step 1: Run STransUNet
        # ----------------------------------
        if img1.shape[0] == 256 and img1.shape[1] == 256:
            final_binary_mask, final_changed_rgb, total_change_percent = \
                self.stransunet.run_on_pair(img1, img2)

            # ✅ Run Mask R-CNN on full image + mask
            maskrcnn_results = self.maskrcnn.run_on_pair(img2, final_binary_mask)

        else:
            # Split images into tiles
            tiles1, grid_size, original_size = split_image_to_tiles(before_img_path, TILE_FOLDER, tile_size=256)
            tiles2, _, _ = split_image_to_tiles(after_img_path, TILE_FOLDER, tile_size=256)

            # Read tiles
            img1_tiles = [cv2.imread(t) for t in tiles1]
            img2_tiles = [cv2.imread(t) for t in tiles2]

            # Run on tiles (STransUNet)
            binary_masks, changed_rgbs, change_percents = self.stransunet.run_on_tiles(img1_tiles, img2_tiles)

            # Save predicted tiles temporarily
            mask_tile_dir = os.path.join(TILE_FOLDER, "mask_tiles")
            rgb_tile_dir = os.path.join(TILE_FOLDER, "rgb_tiles")
            os.makedirs(mask_tile_dir, exist_ok=True)
            os.makedirs(rgb_tile_dir, exist_ok=True)
            cleanup_folder(mask_tile_dir)
            cleanup_folder(rgb_tile_dir)

            for idx, (mask, rgb) in enumerate(zip(binary_masks, changed_rgbs)):
                row = idx // grid_size[1]
                col = idx % grid_size[1]
                cv2.imwrite(os.path.join(mask_tile_dir, f"{row:03d}_{col:03d}.png"), mask * 255)
                cv2.imwrite(os.path.join(rgb_tile_dir, f"{row:03d}_{col:03d}.png"), rgb)

            # Merge tile predictions
            final_binary_mask = merge_tiles_to_image_from_folder(mask_tile_dir, grid_size, original_size, tile_size=256)
            final_changed_rgb = merge_tiles_to_image_from_folder(rgb_tile_dir, grid_size, original_size, tile_size=256)

            # Global % change (based on merged mask)
            total_change_percent = float(np.sum(final_binary_mask > 0) / final_binary_mask.size * 100)

            # ✅ Run Mask R-CNN on tiles
            maskrcnn_results_list = self.maskrcnn.run_on_tiles(img2_tiles, binary_masks)

            # Merge Mask R-CNN results across tiles
            class_counts = {}
            class_pixels = {} 
            class_percentages_global = {}
            vis_imgs = []

            for res in maskrcnn_results_list:
                # aggregate counts
                for cls, cnt in res["counts"].items():
                    class_counts[cls] = class_counts.get(cls, 0) + cnt
                # aggregate % (area-based, so we sum areas then divide later if needed)
                for cls, pct in res["change_percent"].items():
                    class_pixels[cls] = class_pixels.get(cls, 0) 
                vis_imgs.append(res["vis_image"])

            # stitch vis images back to full image
            vis_img = merge_tiles_to_image_from_list(vis_imgs, grid_size, original_size, tile_size=256)
            # total changed pixels across whole image (from merged mask)
            total_changed_pixels = np.sum(final_binary_mask > 0)

            # compute global class percentages (relative to total changed area)
            for cls, pixels in class_pixels.items():
                class_percentages_global[cls] = (pixels / total_changed_pixels * 100) if total_changed_pixels > 0 else 0
            class_percentages = float(np.sum(list(class_percentages_global.values())) > 0)
            return (
                final_binary_mask,    # binary mask (0/255)
                final_changed_rgb,    # changed RGB
                total_change_percent, # % total change
                class_counts,         # number of each detected class
                class_percentages,    # % per class (relative to changed area)
                vis_img               # visualized detections
            )

        # ------------------------------
        # If not tiled case (single pair)
        # ------------------------------
        class_counts = maskrcnn_results["counts"]
        class_percentages = maskrcnn_results["change_percent"]
        vis_img = maskrcnn_results["vis_image"]

        return (
            final_binary_mask*255,    # binary mask (0/255)
            final_changed_rgb,    # changed RGB
            total_change_percent, # % total change
            class_counts,         # number of each detected class
            class_percentages,    # % per class (relative to changed area)
            vis_img               # visualized detections
        )
