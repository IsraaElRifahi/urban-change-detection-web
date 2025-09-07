import os
import cv2
import uuid
import shutil
from utils.image_splitter import split_image_to_tiles

# Define default folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
TILE_FOLDER = "temp_tiles"

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, folder=UPLOAD_FOLDER) -> str:
    """
    Save uploaded file to the uploads folder with a unique name.
    Returns the full path.
    """
    if not allowed_file(file.filename):
        raise ValueError(f"❌ File type not allowed: {file.filename}")

    os.makedirs(folder, exist_ok=True)

    # Generate unique filename
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(folder, filename)

    file.save(file_path)
    return file_path


def prepare_image_for_model(image_path: str, tile_size=256) -> tuple:
    """
    Check image size, split if needed, and return:
    - List of tile paths
    - Grid size (rows, cols)
    - Original image size (height, width)
    """
    # Load image to get original size
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")
    original_size = image.shape[:2]  # (height, width)

    # Split into tiles
    tiles, grid_size = split_image_to_tiles(image_path, TILE_FOLDER, tile_size)
    return tiles, grid_size, original_size


def cleanup_folder(folder: str):
    """Delete all contents of a folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


def get_result_path(filename: str, folder=RESULT_FOLDER) -> str:
    """Return a path to save the final output image."""
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)
