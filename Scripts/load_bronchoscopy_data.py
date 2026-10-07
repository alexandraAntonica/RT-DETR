# Load dataset of annotated bronchoscopy images and convert to COCO-format JSON per split
import glob
import json
import os
import random
import shutil
from collections import defaultdict

from PIL import Image

DATA_ROOT = r"C:\Users\AlexandraAntonica\AtlasInteractiveSR\annotated_data"
SAVE_DIR = r"C:\Users\AlexandraAntonica\tunnel-travelling-experiments\RT-DETR\bronchoscopy_data"

id2label = {0: "Lumen"}
label2id = {v: k for k, v in id2label.items()}
CATEGORIES = [{"id": cat_id, "name": name} for cat_id, name in id2label.items()]


def load_route_items(route_dir):
    """Read a route's COCO annotations.json and resolve each image to its on-disk path."""
    with open(os.path.join(route_dir, "annotations.json")) as f:
        coco = json.load(f)

    coco_id_to_label = {c["id"]: label2id[c["name"]] for c in coco["categories"]}

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    images_dir = os.path.join(route_dir, "images")
    route_name = os.path.basename(route_dir.rstrip(os.sep))
    items = []
    for image in coco["images"]:
        # file_name says .jpg, but frames are stored on disk as .png
        stem = os.path.splitext(image["file_name"])[0]
        image_path = os.path.join(images_dir, stem + ".png")
        if not os.path.exists(image_path):
            continue
        width, height = image.get("width"), image.get("height")
        if width is None or height is None:
            width, height = Image.open(image_path).size
        anns = annotations_by_image[image["id"]]
        items.append({
            "image_path": image_path,
            "dest_name": f"{route_name}__{stem}.png",  # unique name across routes
            "width": width,
            "height": height,
            "boxes": [ann["bbox"] for ann in anns],  # COCO: [x, y, width, height]
            "categories": [coco_id_to_label[ann["category_id"]] for ann in anns],
        })
    return items


def build_coco_split(routes, images_out_dir):
    """Copy images for a split and assemble a COCO-format {images, annotations, categories} dict."""
    os.makedirs(images_out_dir, exist_ok=True)
    images, annotations = [], []
    image_id = 0
    ann_id = 0
    for route_dir in routes:
        for item in load_route_items(route_dir):
            dest_path = os.path.join(images_out_dir, item["dest_name"])
            shutil.copy2(item["image_path"], dest_path)

            images.append({
                "id": image_id,
                "file_name": item["dest_name"],
                "width": item["width"],
                "height": item["height"],
            })
            for bbox, category_id in zip(item["boxes"], item["categories"]):
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })
                ann_id += 1
            image_id += 1

    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def split_routes_and_save(DATA_ROOT, SAVE_DIR):
    route_dirs = sorted(d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d))
    random.Random(1337).shuffle(route_dirs)

    n_val = max(1, round(len(route_dirs) * 0.15))
    n_test = max(1, round(len(route_dirs) * 0.10))
    validation_routes = route_dirs[:n_val]
    test_routes = route_dirs[n_val:n_val + n_test]
    train_routes = route_dirs[n_val + n_test:]

    os.makedirs(SAVE_DIR, exist_ok=True)
    train_coco = build_coco_split(train_routes, os.path.join(SAVE_DIR, "train", "images"))
    validation_coco = build_coco_split(validation_routes, os.path.join(SAVE_DIR, "val", "images"))
    test_coco = build_coco_split(test_routes, os.path.join(SAVE_DIR, "test", "images"))

    # Save the splits to disk
    with open(os.path.join(SAVE_DIR, "train_samples.json"), "w") as f:
        json.dump(train_coco, f)
    with open(os.path.join(SAVE_DIR, "validation_samples.json"), "w") as f:
        json.dump(validation_coco, f)
    with open(os.path.join(SAVE_DIR, "test_samples.json"), "w") as f:
        json.dump(test_coco, f)

    return train_coco, validation_coco, test_coco

split_routes_and_save(DATA_ROOT, SAVE_DIR)

