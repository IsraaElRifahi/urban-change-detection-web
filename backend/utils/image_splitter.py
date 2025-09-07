import os
import cv2
import shutil
import numpy as np

def merge_tiles_to_image_from_list(tiles, grid_size, original_size, tile_size=256):
    """
    Merge tiles from a list of numpy arrays into one image.
    """
    rows, cols = grid_size
    if not tiles:
        raise ValueError("❌ No tiles provided to merge.")

    merged_height = rows * tile_size
    merged_width = cols * tile_size
    merged_image = np.zeros((merged_height, merged_width, 3), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(tiles):
                break
            merged_image[r*tile_size:(r+1)*tile_size,
                         c*tile_size:(c+1)*tile_size] = tiles[idx]
            idx += 1

    orig_height, orig_width = original_size
    return merged_image[:orig_height, :orig_width]


def merge_tiles_to_image_from_folder(tiles_folder, grid_size, original_size, tile_size=256):
    """
    Merge tiles from a folder (disk-based workflow).
    """
    rows, cols = grid_size
    tile_paths = sorted(os.listdir(tiles_folder))
    if not tile_paths:
        raise ValueError(f"❌ No tiles found in {tiles_folder}")

    merged_height = rows * tile_size
    merged_width = cols * tile_size
    merged_image = np.zeros((merged_height, merged_width, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            tile_name = f"{r:03d}_{c:03d}.png"
            tile_path = os.path.join(tiles_folder, tile_name)
            if not os.path.exists(tile_path):
                raise FileNotFoundError(f"Tile not found: {tile_path}")

            tile = cv2.imread(tile_path)
            merged_image[r*tile_size:(r+1)*tile_size,
                         c*tile_size:(c+1)*tile_size] = tile

    orig_height, orig_width = original_size
    return merged_image[:orig_height, :orig_width]

def split_image_to_tiles(image_path, output_folder, tile_size=256):
    """
    Splits an image into tiles of size tile_size × tile_size.
    If the image is already tile_size × tile_size, returns it directly.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder where tiles will be saved.
        tile_size (int): Size of each square tile.

    Returns:
        tiles (list): List of tile file paths.
        grid_size (tuple): (rows, cols) for later merging.
        original_size (tuple): (height, width) of the original image before padding.
    """
    # Remove old tiles
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")

    original_height, original_width = image.shape[:2]   # <-- keep original size

    # If image is exactly tile_size × tile_size → no need to split
    if original_height == tile_size and original_width == tile_size:
        tile_path = os.path.join(output_folder, "000_000.png")
        cv2.imwrite(tile_path, image)
        return [tile_path], (1, 1), (original_height, original_width)

    # If image is not divisible by tile_size → pad with black pixels
    pad_bottom = (tile_size - (original_height % tile_size)) % tile_size
    pad_right = (tile_size - (original_width % tile_size)) % tile_size
    if pad_bottom > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

    padded_height, padded_width = image.shape[:2]
    rows = padded_height // tile_size
    cols = padded_width // tile_size
    tiles = []

    # Split into tiles (keep original orientation)
    for r in range(rows):
        for c in range(cols):
            tile = image[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]
            tile_name = f"{r:03d}_{c:03d}.png"
            tile_path = os.path.join(output_folder, tile_name)
            cv2.imwrite(tile_path, tile)
            tiles.append(tile_path)

    return tiles, (rows, cols), (original_height, original_width)   # <-- now 3 values

