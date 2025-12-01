import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from segment_anything import sam_model_registry
from dataset import BuildingDataset
import os
from tqdm import tqdm

# --- Configuration ---
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_hq_vit_b.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 1
OUTPUT_DIR = "checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper: IoU Calculation ---
def calculate_iou(pred_mask, true_mask):
    # Pred mask is raw logits, convert to binary
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    if union == 0:
        return 1.0 # Both empty
    return intersection / union

# --- Loss Function ---
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_masks, true_masks):
        loss_bce = self.bce(pred_masks, true_masks)
        pred_probs = torch.sigmoid(pred_masks)
        smooth = 1e-5
        intersection = (pred_probs * true_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + true_masks.sum(dim=(2, 3))
        loss_dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return loss_bce + loss_dice.mean()

# --- 2. Setup Model ---
print(f"Loading SAM-HQ ({MODEL_TYPE}) from {CHECKPOINT_PATH}...")
# Workaround: Pre-load checkpoint with map_location to handle GPU→CPU transfer
checkpoint_state = torch.load(CHECKPOINT_PATH, map_location=device)
# Build model without checkpoint first
sam_model = sam_model_registry[MODEL_TYPE](checkpoint=None)
# Load only matching keys (ignore SAM-HQ specific layers if checkpoint has them)
model_dict = sam_model.state_dict()
checkpoint_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
sam_model.load_state_dict(checkpoint_dict, strict=False)
sam_model.to(device)

# --- 3. Freeze Components ---
# We only want to train the Mask Decoder.
# Freezing the Image Encoder (ViT) saves massive amounts of memory.
for name, param in sam_model.named_parameters():
    if "image_encoder" in name or "prompt_encoder" in name:
        param.requires_grad = False
    else:
        # This unfreezes the mask_decoder and any HQ-specific layers
        param.requires_grad = True

# Optimizer only updates parameters that require grad
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=LEARNING_RATE)
loss_fn = CombinedLoss()


# Freeze Encoder
for name, param in sam_model.named_parameters():
    if "image_encoder" in name or "prompt_encoder" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=LEARNING_RATE)
loss_fn = CombinedLoss()

# --- Data Split (Train vs Val) ---
full_dataset = BuildingDataset(root_dir="sam_tiles")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

print(f"Splitting data: {train_size} Training, {val_size} Validation")
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Training Loop ---
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    # --- TRAIN STEP ---
    sam_model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        boxes = batch['box'].to(device)

        if len(boxes.shape) == 2: boxes = boxes.unsqueeze(1)
        if images.shape[-1] != 1024:
            input_images = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            input_images = images

        # FIX 1: Capture 'interm_embeddings' from the encoder
        with torch.no_grad():
            image_embeddings, interm_embeddings = sam_model.image_encoder(input_images)

        scale_x, scale_y = 1024 / images.shape[3], 1024 / images.shape[2]
        boxes_scaled = boxes.clone()
        boxes_scaled[:, :, 0] *= scale_x
        boxes_scaled[:, :, 2] *= scale_x
        boxes_scaled[:, :, 1] *= scale_y
        boxes_scaled[:, :, 3] *= scale_y

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None, boxes=boxes_scaled, masks=None
            )

        # FIX 2: Pass 'interm_embeddings' and 'hq_token_only' to the decoder
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,  # <--- ADDED
            interm_embeddings=interm_embeddings  # <--- ADDED
        )

        upscaled_masks = F.interpolate(low_res_masks, size=(images.shape[2], images.shape[3]), mode="bilinear",
                                       align_corners=False)
        loss = loss_fn(upscaled_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- VALIDATION STEP ---
    sam_model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['box'].to(device)

            if len(boxes.shape) == 2: boxes = boxes.unsqueeze(1)
            if images.shape[-1] != 1024:
                input_images = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
            else:
                input_images = images

            # FIX 1 (Validation): Capture interm_embeddings
            image_embeddings, interm_embeddings = sam_model.image_encoder(input_images)

            scale_x, scale_y = 1024 / images.shape[3], 1024 / images.shape[2]
            boxes_scaled = boxes.clone()
            boxes_scaled[:, :, 0] *= scale_x
            boxes_scaled[:, :, 2] *= scale_x
            boxes_scaled[:, :, 1] *= scale_y
            boxes_scaled[:, :, 3] *= scale_y

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None, boxes=boxes_scaled, masks=None
            )

            # FIX 2 (Validation): Pass arguments
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,  # <--- ADDED
                interm_embeddings=interm_embeddings  # <--- ADDED
            )

            upscaled_masks = F.interpolate(low_res_masks, size=(images.shape[2], images.shape[3]), mode="bilinear",
                                           align_corners=False)

            val_loss += loss_fn(upscaled_masks, masks).item()
            val_iou += calculate_iou(upscaled_masks, masks).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)

    print(
        f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val IoU={avg_val_iou:.4f}")

    torch.save(sam_model.state_dict(), os.path.join(OUTPUT_DIR, f"sam_hq_epoch_{epoch + 1}.pth"))