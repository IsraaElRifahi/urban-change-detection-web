import cv2
import numpy as np
import os

def extract_changed_regions(change_mask: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
    """
    Extract RGB regions corresponding to changed areas based on the binary change mask.
    """
    if change_mask.ndim != 2:
        raise ValueError("Change mask must be a 2D binary image.")
    if rgb_image.shape[:2] != change_mask.shape:
        raise ValueError("RGB image and mask must have the same height and width.")

    binary_mask = (change_mask > 127).astype(np.uint8)
    mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    changed_rgb = rgb_image * mask_3ch
    return changed_rgb

def calculate_change_percentage(binary_mask: np.ndarray) -> float:
    """
    Calculate the percentage of changed pixels in the binary mask.

    Args:
        change_mask (np.ndarray): 2D binary image (0 = no change, 255 = change).

    Returns:
        float: Percentage of changed area with respect to the whole image.
    """
    total_pixels = binary_mask.size
    changed_pixels = np.sum(binary_mask > 0)
    print(f"change-pixel= {changed_pixels}")
    return (changed_pixels / total_pixels) * 100

def save_image(image: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)