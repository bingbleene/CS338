import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ===== PATH =====
COCO_DIR = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"

DATASETS = {
    "refcoco": "/kaggle/input/datasets/kenji0901/refcoco/anns/refcoco/train.json",
    "refcoco+": "/kaggle/input/datasets/kenji0901/refcoco-plus/anns/refcoco+/train.json",
    "refcocog_u": "/kaggle/input/datasets/kenji0901/refcocog-u/anns/refcocog_u/train.json",
}

def get_text(sample):
    if "sentences" in sample:
        if isinstance(sample["sentences"][0], dict):
            return sample["sentences"][0].get("sent", "NO TEXT")
        else:
            return sample["sentences"][0]
    return "NO TEXT"

def load_dataset(name, ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    return data

def preprocess_image(img_path, input_size=416):
    """
    Load and preprocess image without spike encoding.
    SNNVisionEncoder will do rate encoding internally.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img

def get_sample(data, idx, input_size=416):
    """
    Get a sample without pre-encoding spikes.
    SNNVisionEncoder.rate_encode() will handle spike generation.
    """
    sample = data[idx]
    img_name = sample.get("img_name")
    if img_name is None:
        return None, None, None

    img_path = os.path.join(COCO_DIR, img_name)
    img = preprocess_image(img_path, input_size)
    if img is None:
        return None, None, None

    text = get_text(sample)
    bbox = sample.get("bbox")
    
    # Return normalized image (SNNVisionEncoder will encode to spikes)
    return img, text, bbox

# ===== Example Usage =====
if __name__ == "__main__":
    data = load_dataset("refcoco", DATASETS["refcoco"])
    sample_idx = random.randint(0, len(data)-1)
    img, text, bbox = get_sample(data, sample_idx)
    if img is not None:
        print(f"Image shape: {img.shape}")  # Should be (3, 416, 416)
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Text: {text}")
        print(f"BBox: {bbox}")
        print("\nNote: SNNVisionEncoder will encode image to spikes internally using rate coding")
    else:
        print("Failed to load sample")