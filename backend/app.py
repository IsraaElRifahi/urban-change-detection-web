import logging
from flask import Flask, request, jsonify, send_file, send_from_directory
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from flask_cors import CORS
import os
import cv2
import numpy as np
import uuid
import shutil
#from inference.runner_stransunet import STransUNetRunner
from inference.pipeline import ChangeDetectionPipeline   # ✅ use your pipeline

# ====== Logging setup ======
logging.basicConfig(
    level=logging.DEBUG,  # show everything
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # write logs to file
        logging.StreamHandler()          # also print logs to console
    ]
)
logger = logging.getLogger(__name__)
# ===========================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
# ======= Config =======
#MODEL_FOLDER = "models"
#UPLOAD_FOLDER = "uploads"
#RESULT_FOLDER = "results"
#TILE_FOLDER = "temp_tiles"
#TILE_SIZE = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
TILE_FOLDER = os.path.join(BASE_DIR, "temp_tiles")
TILE_SIZE = 256
# .pth files mapping
MODEL_PATHS = {
    "damage": {
        "stransunet": os.path.join(MODEL_FOLDER, "damage.pth"),
        "maskrcnn": os.path.join(MODEL_FOLDER, "MaskRCNN.pth"),
    },
    "building": {
        "stransunet": os.path.join(MODEL_FOLDER, "new_building.pth"),
        "maskrcnn": os.path.join(MODEL_FOLDER, "MaskRCNN.pth"),
    }
}

# ======= Helpers =======
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "tif", "tiff"}

def save_uploaded_file(file) -> str:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

def cleanup_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    before_file = request.files.get("before")
    after_file = request.files.get("after")
    model_type = request.form.get("mode")

    if not before_file or not after_file:
        return jsonify({"error": "Both before and after images are required"}), 400
    if model_type not in MODEL_PATHS:
        return jsonify({"error": "Invalid detection mode"}), 400

    before_path = save_uploaded_file(before_file)
    after_path = save_uploaded_file(after_file)

     # Initialize pipeline with both models
    pipeline = ChangeDetectionPipeline(
        stransunet_model_path=MODEL_PATHS[model_type]["stransunet"],
        maskrcnn_weight_path=MODEL_PATHS[model_type]["maskrcnn"]
    )

    # Run pipeline
    mask, rgb, change_percent, class_counts, class_percentages, vis_img = pipeline.run(before_path, after_path)
    # Ensure mask is 0/255 uint8
    if mask.dtype != "uint8":
        mask = (mask > 0).astype("uint8") * 255
    # Save results
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    mask_path = os.path.join(RESULT_FOLDER, "binary_mask.png")
    rgb_path = os.path.join(RESULT_FOLDER, "changed_rgb.png")
    vis_path = os.path.join(RESULT_FOLDER, "vis_maskrcnn.png")
    # ensure 3 channels for browser display
    if len(mask.shape) == 2:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_rgb = mask

    cv2.imwrite(mask_path, mask_rgb)

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(vis_path, vis_img)

    return jsonify({
        "mask_url": f"/download?file=binary_mask.png",
        "rgb_url": f"/download?file=changed_rgb.png",
        "vis_url": f"/download?file=vis_maskrcnn.png",
        "change_percent": change_percent,
        "class_counts": class_counts,
        "class_percentages": class_percentages
    })
@app.route("/download", methods=["GET"])
def download_image():
    file = request.args.get("file")
    if not file:
        return {"error": "File parameter missing"}, 400
    
    file_path = os.path.join(RESULT_FOLDER, file)
    if not os.path.exists(file_path):
        return {"error": "File not found"}, 404
    
    # Serve as image (no attachment)
    return send_from_directory(RESULT_FOLDER, file)

@app.route("/download_pdf", methods=["GET"])
def download_file():
    # Collect result data
    mask_url = os.path.join(RESULT_FOLDER, "binary_mask.png")
    rgb_url = os.path.join(RESULT_FOLDER, "changed_rgb.png")
    vis_url = os.path.join(RESULT_FOLDER, "vis_maskrcnn.png")

    # Load metadata
    import json
    meta_path = os.path.join(RESULT_FOLDER, "results_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"change_percent": "N/A", "class_counts": {}, "class_percentages": {}}

    change_percent = metadata.get("change_percent", "N/A")
    class_counts = metadata.get("class_counts", {})
    class_percentages = metadata.get("class_percentages", {})

    pdf_path = os.path.join(RESULT_FOLDER, "report.pdf")

    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Add text
    elements.append(Paragraph("<b>Urban Change Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Change Percentage: {change_percent}", styles["Normal"]))
    elements.append(Paragraph(f"Class Counts: {class_counts}", styles["Normal"]))
    elements.append(Paragraph(f"Class Percentages: {class_percentages}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Add images (scaled to fit one page)
    max_width, max_height = 250, 250  # smaller to fit multiple on one page
    if os.path.exists(mask_url):
        elements.append(Paragraph("Binary Mask:", styles["Heading2"]))
        elements.append(Image(mask_url, width=max_width, height=max_height))

    if os.path.exists(rgb_url):
        elements.append(Paragraph("Changed RGB:", styles["Heading2"]))
        elements.append(Image(rgb_url, width=max_width, height=max_height))

    if os.path.exists(vis_url):
        elements.append(Paragraph("Mask R-CNN Visualization:", styles["Heading2"]))
        elements.append(Image(vis_url, width=max_width, height=max_height))

    doc.build(elements)

    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    else:
        return {"error": "PDF not generated"}, 500



if __name__ == "__main__":
    app.run(debug=True)
