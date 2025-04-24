import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

SCENE_DIR = "./data/private/scenes"
MASK_COLOR = (135, 206, 235)  # RGB of the mask region
PADDING = 10  # Pixel padding when cropping

def extract_mask_and_crop(image):
    image_np = np.array(image)

    # Create binary mask
    mask = np.all(image_np == MASK_COLOR, axis=-1).astype(np.uint8)

    # Denoising: keep only the largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return Image.fromarray(mask * 255), None

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_cc = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8) * 255

    # Find bounding box
    ys, xs = np.where(largest_cc == 255)
    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(largest_cc), None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding
    W, H = image.size
    x_min = max(x_min - PADDING, 0)
    x_max = min(x_max + PADDING, W - 1)
    y_min = max(y_min - PADDING, 0)
    y_max = min(y_max + PADDING, H - 1)

    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    return Image.fromarray(largest_cc), cropped

def process_all_scenes():
    for scene_id in tqdm(sorted(os.listdir(SCENE_DIR)), desc="Processing scenes"):
        scene_path = os.path.join(SCENE_DIR, scene_id)
        input_path = os.path.join(scene_path, "masked.png")
        mask_output = os.path.join(scene_path, "mask.png")
        crop_output = os.path.join(scene_path, "crop.png")
        try:
            image = Image.open(input_path).convert("RGB")
            mask_img, crop_img = extract_mask_and_crop(image)

            mask_img.save(mask_output)
            if crop_img:
                crop_img.save(crop_output)
        # Used for .DS_Store files on MacOS
        except NotADirectoryError:
            print(f"Directory {scene_path} does not exist. Skipping.")

if __name__ == '__main__':
    process_all_scenes()
    print("✅ Done. Cleaned mask.png and crop.png saved to each scene.")
