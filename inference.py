import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/sam_hq_epoch_7.pth"  # Your best epoch
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your NEW large drone image
IMAGE_PATH = "input_files/inference_drone02.tif"  # Change this to your new file
OUTPUT_PATH = "prediction_result03.png"

# Tiling settings
TILE_SIZE = 500  # SAM native size
OVERLAP = 0  # Overlap in pixels (increase if you see cut-off buildings at edges)

# --- 1. LOAD MODEL ---
print(f"Loading model from {CHECKPOINT_PATH}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)

# Initialize the Automatic Mask Generator
# This wrapper handles the HQ-features internally, so we don't need manual loop adjustments here.
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,  # Grid 32x32 points. Increase if buildings are tiny.
    pred_iou_thresh=0.8,  # Filter out low quality masks
    stability_score_thresh=0.85,
    crop_n_layers=0,  # Keep simple for now
    min_mask_region_area=100,  # Ignore tiny noise blobs
)

# --- 2. LOAD IMAGE ---
print(f"Loading image: {IMAGE_PATH}")
large_image = cv2.imread(IMAGE_PATH)
if large_image is None:
    raise FileNotFoundError(f"Could not open {IMAGE_PATH}")

# Convert to RGB
large_image_rgb = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)
H, W, _ = large_image.shape

# Create an empty black mask for the result
final_mask = np.zeros((H, W), dtype=np.uint8)

print(f"Image Size: {W}x{H}. Starting sliding window inference...")

# --- 3. SLIDING WINDOW LOOP ---
# Calculate steps
step = TILE_SIZE - OVERLAP
rows = range(0, H, step)
cols = range(0, W, step)

total_steps = len(rows) * len(cols)
pbar = tqdm(total=total_steps)

for y in rows:
    for x in cols:
        # Define window coordinates
        y2 = min(y + TILE_SIZE, H)
        x2 = min(x + TILE_SIZE, W)

        # Real width/height of this tile (might be smaller at edges)
        h_crop = y2 - y
        w_crop = x2 - x

        # Crop the image
        crop = large_image_rgb[y:y2, x:x2]

        # Run SAM-HQ Inference
        # Returns a list of dicts: [{'segmentation': numpy_mask, 'area': ...}, ...]
        masks = mask_generator.generate(crop)

        # Merge all detected masks in this tile into one binary layer
        tile_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)

        for mask_data in masks:
            # mask_data['segmentation'] is True/False boolean array
            m = mask_data['segmentation'].astype(np.uint8)
            tile_mask = np.maximum(tile_mask, m)  # Combine (Logical OR)

        # Paste the result into the global mask
        # Note: If using overlap, simply overwriting works for binary,
        # but max() is safer to preserve detections from previous overlaps.
        current_region = final_mask[y:y2, x:x2]
        final_mask[y:y2, x:x2] = np.maximum(current_region, tile_mask)

        pbar.update(1)

pbar.close()

# --- 4. SAVE RESULT ---
# Multiply by 255 to make it visible (0=Black, 255=White)
cv2.imwrite(OUTPUT_PATH, final_mask * 255)
print(f"Done! Prediction saved to {OUTPUT_PATH}")