import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

SCENE_DIR = "private_data/scenes"
MASK_COLOR = (135, 206, 235)  # RGB của vùng mask
PADDING = 10  # Pixel padding khi crop

def extract_mask_and_crop(image):
    image_np = np.array(image)

    # Tạo mask nhị phân
    mask = np.all(image_np == MASK_COLOR, axis=-1).astype(np.uint8)

    # Lọc nhiễu: giữ lại connected component lớn nhất
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return Image.fromarray(mask * 255), None

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_cc = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8) * 255

    # Tìm bounding box
    ys, xs = np.where(largest_cc == 255)
    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(largest_cc), None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Thêm padding
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

        image = Image.open(input_path).convert("RGB")
        mask_img, crop_img = extract_mask_and_crop(image)

        mask_img.save(mask_output)
        if crop_img:
            crop_img.save(crop_output)

if __name__ == '__main__':
    process_all_scenes()
    print("✅ Done. Cleaned mask.png and crop.png saved to each scene.")
