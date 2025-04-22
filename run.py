import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from rembg import remove

# Config
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
SCENE_DIR = "private_data/scenes"
OBJECT_DIR = "private_data/objects"
OUTPUT_DIR = "output_new"
MASK_COLOR = (135, 206, 235)

# Load CLIP model
# clip_model_name = "openai/clip-vit-large-patch14"
clip_model_name = "openai/clip-vit-large-patch14-224"  # Use the larger model for better performance
model = CLIPModel.from_pretrained(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)

# Image transform
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor()
])

def cosine_similarity_np(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def extract_mask_only(image: Image.Image):
    image_np = np.array(image)
    mask = np.all(image_np == MASK_COLOR, axis=-1).astype(np.uint8) * 255
    return Image.fromarray(mask)

def encode_all_objects():
    obj_feats = {}
    obj_feats_binary = {}
    for obj_id in tqdm(sorted(os.listdir(OBJECT_DIR)), desc="Encoding objects"):
        img_path = os.path.join(OBJECT_DIR, obj_id, "image.jpg")
        image = Image.open(img_path).convert("RGB")

        image_binary = remove(image)
        image_binary = np.array(image_binary)
        image_binary = (image_binary > 0).astype(np.uint8) * 255
        image_binary = Image.fromarray(image_binary)

        image_binary.save("debug.png")

        inputs = processor(images=image, return_tensors="pt").to(device)
        inputs_binary = processor(images=image_binary, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs).cpu().numpy().squeeze()
            feat_binary = model.get_image_features(**inputs_binary).cpu().numpy().squeeze()

        obj_feats[obj_id] = feat
        obj_feats_binary[obj_id] = feat_binary
    return obj_feats, obj_feats_binary

def match_by_shape(scene_id, obj_feats_binary):
    mask_img = Image.open(os.path.join(SCENE_DIR, scene_id, "crop.png")).convert("RGB")
    mask_img = extract_mask_only(mask_img)
    inputs = processor(images=mask_img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs).cpu().numpy().squeeze()

    return sorted(obj_feats_binary, key=lambda obj_id: cosine_similarity_np(feat, obj_feats_binary[obj_id]), reverse=True)

def match_by_text(scene_id, obj_feats):
    with open(os.path.join(SCENE_DIR, scene_id, "query.txt")) as f:
        prompt = f.read().strip()

    inputs = processor(text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs).cpu().numpy().squeeze()

    return sorted(obj_feats, key=lambda obj_id: cosine_similarity_np(text_feat, obj_feats[obj_id]), reverse=True)

def match_by_text_then_shape(scene_id, obj_feats, obj_feats_binary):
    text_results = match_by_text(scene_id, obj_feats)[:15]

    mask_img = Image.open(os.path.join(SCENE_DIR, scene_id, "crop.png")).convert("RGB")
    mask_img = extract_mask_only(mask_img)
    inputs = processor(images=mask_img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = model.get_image_features(**inputs).cpu().numpy().squeeze()

    filtered_results = {obj_id: cosine_similarity_np(img_feat, obj_feats_binary[obj_id]) for obj_id in text_results}
    return sorted(filtered_results, key=filtered_results.get, reverse=True)[:10]

def match_by_text_then_shape_order_by_text(scene_id, obj_feats, obj_feats_binary):
    # Step 1: Get top-15 by text
    text_results = match_by_text(scene_id, obj_feats)[:15]

    # Step 2: Compute shape similarity
    mask_img = Image.open(os.path.join(SCENE_DIR, scene_id, "crop.png")).convert("RGB")
    mask_img = extract_mask_only(mask_img)
    inputs = processor(images=mask_img, return_tensors="pt").to(device)
    with torch.no_grad():
        shape_feat = model.get_image_features(**inputs).cpu().numpy().squeeze()

    # Compute shape similarity within top-15
    shape_scores = {
        obj_id: cosine_similarity_np(shape_feat, obj_feats_binary[obj_id]) for obj_id in text_results
    }

    # Step 3: Select top-10 by shape
    top10_by_shape = sorted(shape_scores.items(), key=lambda x: -x[1])[:10]
    top10_obj_ids = [obj_id for obj_id, _ in top10_by_shape]

    # Step 4: Order top-10 by text similarity
    text_scores = {
        obj_id: cosine_similarity_np(obj_feats[obj_id], obj_feats[obj_id])  # dummy to use obj_feats[obj_id] below
        for obj_id in top10_obj_ids
    }
    query_prompt = open(os.path.join(SCENE_DIR, scene_id, "query.txt")).read().strip()
    text_inputs = processor(text=query_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        query_text_feat = model.get_text_features(**text_inputs).cpu().numpy().squeeze()
    for obj_id in top10_obj_ids:
        text_scores[obj_id] = cosine_similarity_np(query_text_feat, obj_feats[obj_id])

    return sorted(top10_obj_ids, key=lambda obj_id: -text_scores[obj_id])


def major_voting(scene_ids, obj_feats, obj_feats_binary):
    result = {}
    for scene_id in tqdm(scene_ids, desc="Voting"):
        counter = defaultdict(int)

        for obj_id in match_by_text(scene_id, obj_feats)[:10]:
            counter[obj_id] += 2

        for obj_id in match_by_text_then_shape_order_by_text(scene_id, obj_feats, obj_feats_binary):
            counter[obj_id] += 1

        for obj_id in match_by_shape(scene_id, obj_feats_binary)[:10]:
            counter[obj_id] += 1

        for obj_id in match_by_text_then_shape(scene_id, obj_feats, obj_feats_binary):
            counter[obj_id] += 1

        sorted_objs = sorted(counter.items(), key=lambda x: -x[1])
        result[scene_id] = [obj for obj, _ in sorted_objs[:10]]
    return result

def save_submission_csv(results, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for scene_id, obj_ids in results.items():
            writer.writerow([scene_id] + obj_ids)

def save_visual_outputs(method_name, scene_ids, topk_results):
    for scene_id in scene_ids:
        out_dir = os.path.join(OUTPUT_DIR, method_name, scene_id)
        os.makedirs(out_dir, exist_ok=True)

        for rank, obj_id in enumerate(topk_results[scene_id]):
            obj_img_path = os.path.join(OBJECT_DIR, obj_id, "image.jpg")
            obj_img = Image.open(obj_img_path).convert("RGB")
            save_image(obj_img, os.path.join(out_dir, f"{rank+1:02d}_{obj_id}.jpg"))

if __name__ == '__main__':
    scene_ids = sorted(os.listdir(SCENE_DIR))
    obj_feats, obj_feats_binary = encode_all_objects()

    result_shape = {sid: match_by_shape(sid, obj_feats_binary)[:10] for sid in scene_ids}
    save_submission_csv(result_shape, "selab_shape_only.csv")
    save_visual_outputs("shape_only", scene_ids, result_shape)

    result_text = {sid: match_by_text(sid, obj_feats)[:10] for sid in scene_ids}
    save_submission_csv(result_text, "selab_text_only.csv")
    save_visual_outputs("text_only", scene_ids, result_text)

    result_text_shape = {sid: match_by_text_then_shape(sid, obj_feats, obj_feats_binary) for sid in scene_ids}
    save_submission_csv(result_text_shape, "selab_text_then_shape_by_shape.csv")
    save_visual_outputs("text_then_shape_by_shape", scene_ids, result_text_shape)

    result_text_shape_by_text = {sid: match_by_text_then_shape_order_by_text(sid, obj_feats, obj_feats_binary) for sid in scene_ids}
    save_submission_csv(result_text_shape_by_text, "selab_text_then_shape_by_text.csv")
    save_visual_outputs("text_then_shape_by_text", scene_ids, result_text_shape_by_text)

    result_vote = major_voting(scene_ids, obj_feats, obj_feats_binary)
    save_submission_csv(result_vote, "selab_major_voting.csv")
    save_visual_outputs("major_voting", scene_ids, result_vote)

    print("✅ All strategies complete.")