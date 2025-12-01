import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class BuildingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.transform = transform

        # Get all image filenames
        all_images = sorted(os.listdir(self.image_dir))

        # Filter out empty masks (Pre-scanning)
        print("Scanning dataset to remove empty tiles...")
        self.valid_images = []
        for img_name in tqdm(all_images):
            # -----------------------------------------------------------
            # FIX: Convert 'row_x_col_y_image.tif' -> 'row_x_col_y_mask.tif'
            mask_name = img_name.replace("_image", "_mask")
            # -----------------------------------------------------------

            mask_path = os.path.join(self.mask_dir, mask_name)

            # Verify file exists to prevent crashing later
            if not os.path.exists(mask_path):
                # If the mask file is missing entirely, skip this image
                continue

            # Read mask to check for buildings
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Only keep if there is at least one white pixel (building)
            if mask is not None and np.any(mask > 0):
                self.valid_images.append(img_name)

        print(
            f"Kept {len(self.valid_images)} tiles (dropped {len(all_images) - len(self.valid_images)} empty/missing).")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]

        # 1. Load Image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load Mask
        # -----------------------------------------------------------
        # FIX: Same conversion here
        mask_name = img_name.replace("_image", "_mask")
        # -----------------------------------------------------------
        mask_path = os.path.join(self.mask_dir, mask_name)

        full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        full_mask = (full_mask > 0).astype(np.uint8)  # Ensure binary 0/1

        # 3. Instance Separation
        # Use connected components to find separate buildings
        num_labels, labels_im = cv2.connectedComponents(full_mask)

        # labels_im contains 0 for background, 1 for building A, 2 for building B...
        if num_labels > 1:
            # Pick one building randomly (ignoring background 0)
            choice_idx = np.random.randint(1, num_labels)
            instance_mask = (labels_im == choice_idx).astype(np.uint8)
        else:
            # Should not happen if init filter works, but as fallback:
            instance_mask = full_mask

        # 4. Generate Prompt (Box) for the SELECTED instance
        coords = np.argwhere(instance_mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Optional: Add random noise/perturbation to the box
            perturb = 5
            H, W = instance_mask.shape
            x_min = max(0, x_min - np.random.randint(0, perturb))
            y_min = max(0, y_min - np.random.randint(0, perturb))
            x_max = min(W, x_max + np.random.randint(0, perturb))
            y_max = min(H, y_max + np.random.randint(0, perturb))

            bbox = [x_min, y_min, x_max, y_max]
        else:
            bbox = [0, 0, 1, 1]

        # 5. Prepare Tensors
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.as_tensor(instance_mask[None, :, :]).float()
        box_tensor = torch.as_tensor(bbox).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
            "img_name": img_name
        }
