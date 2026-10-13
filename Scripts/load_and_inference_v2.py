"""Inference with a checkpoint trained by this repo's rtdetrv2_pytorch code (e.g. output/.../best.pth).

Unlike load_and_inference_v1.py (Hugging Face transformers checkpoints), this builds the model
from the training config and loads the .pth weights.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.ops as ops
import torchvision.transforms as T
from PIL import Image, ImageFont, ImageDraw

from filter_contained_bbox import contained_bbox_keep

RTDETR_DIR = Path(__file__).resolve().parent.parent / "rtdetrv2_pytorch"
sys.path.insert(0, str(RTDETR_DIR))
from src.core import YAMLConfig  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# one color per box (by score rank) so a moved label can still be matched to its box;
# all dark enough for white label text, and no reds/oranges so boxes stand out against tissue
BOX_COLORS = ["#2a78d6", "#008300", "#4a3aa7", "#0e7c86"]
# the model has a single class (label 0)
CLASS_NAME = "lumen"


def load_v2(config, checkpoint, device):
    # pretrained=False: no need to fetch ImageNet backbone weights, the checkpoint overwrites them
    cfg = YAMLConfig(config, PResNet={"pretrained": False})

    state = torch.load(checkpoint, map_location="cpu")
    state = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(state)

    model = cfg.model.deploy().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)
    input_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    return model, postprocessor, input_size


def detect_boxes(images, model, postprocessor, input_size, device, threshold=0.3, nms=False, iou_threshold=0.5,
                 filter_contained=False):
    """Runs one batch of PIL images, returns one result dict per image."""
    transforms = T.Compose([T.Resize(input_size), T.ToTensor()])  # same as the val transforms
    inputs = torch.stack([transforms(image) for image in images]).to(device)
    # (w, h) per image: used to rescale boxes to the original image size
    target_sizes = torch.tensor([image.size for image in images], device=device)

    with torch.no_grad():
        outputs = model(inputs)
        batch_results = postprocessor(outputs, target_sizes)

    results = []
    for result in batch_results:
        keep = result["scores"] > threshold
        result = {k: v[keep] for k, v in result.items()}

        if nms:
            keep = ops.batched_nms(result["boxes"], result["scores"], result["labels"], iou_threshold)
            result = {k: v[keep] for k, v in result.items()}

        if filter_contained:
            keep = contained_bbox_keep(result["boxes"].tolist(), result["scores"].tolist())
            result = {k: v[keep] for k, v in result.items()}

        results.append({k: v.cpu() for k, v in result.items()})
    return results


def collect_frames(paths):
    """Expands folders into their image files (sorted by name, i.e. frame order for zero-padded names)."""
    frames = []
    for path in map(Path, paths):
        if path.is_dir():
            frames.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            frames.append(path)
    return frames


def plot_result(image, result):
    image_with_boxes = image.convert("RGB")
    draw = ImageDraw.Draw(image_with_boxes)

    try:
        font = ImageFont.load_default(size=max(12, image_with_boxes.width // 50))
    except TypeError:  # Pillow < 10.1 doesn't accept a size
        font = ImageFont.load_default()

    # highest score first, so the best detection gets the preferred label spot
    order = result["scores"].argsort(descending=True).tolist()
    colors = {i: BOX_COLORS[rank % len(BOX_COLORS)] for rank, i in enumerate(order)}

    # all boxes first, so no box edge is drawn over a label
    for i in order:
        draw.rectangle(result["boxes"][i].tolist(), outline=colors[i], width=2)

    placed = []
    for i in order:
        text = f"{CLASS_NAME} {result['scores'][i].item():.2f}"
        rect, offset = place_label(draw, text, font, result["boxes"][i].tolist(), placed, image_with_boxes.size)
        placed.append(rect)
        draw.rectangle(rect, fill=colors[i])
        draw.text((rect[0] + offset[0], rect[1] + offset[1]), text, fill="white", font=font)

    return image_with_boxes


def place_label(draw, text, font, box, placed, image_size, pad=2):
    """Returns (label rectangle, text offset inside it) at the first spot around `box`
    that stays in the image and doesn't overlap an already placed label."""
    x, y, x2, y2 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    img_w, img_h = image_size

    def clamp(cx, cy):
        cx, cy = min(max(cx, 0), img_w - w), min(max(cy, 0), img_h - h)
        return (cx, cy, cx + w, cy + h)

    def free(rect):
        return not any(rect[0] < p[2] and p[0] < rect[2] and rect[1] < p[3] and p[1] < rect[3] for p in placed)

    # above-left, above-right, inside top-left, inside top-right, below-left, below-right
    candidates = [(x, y - h), (x2 - w, y - h), (x, y), (x2 - w, y), (x, y2), (x2 - w, y2)]
    rect = next((r for r in map(lambda c: clamp(*c), candidates) if free(r)), None)
    if rect is None:
        # everything around the box is taken: stack below the first spot until free
        rect = clamp(*candidates[0])
        while not free(rect) and rect[3] + h <= img_h:
            rect = (rect[0], rect[1] + h, rect[2], rect[3] + h)

    return rect, (pad - left, pad - top)


def main(args):
    frames = collect_frames(args.images)
    if not frames:
        sys.exit("no images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, postprocessor, input_size = load_v2(args.config, args.checkpoint, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = []

    for start in range(0, len(frames), args.batch_size):
        batch = frames[start:start + args.batch_size]
        images = [Image.open(frame).convert("RGB") for frame in batch]
        results = detect_boxes(
            images, model, postprocessor, input_size, device,
            threshold=args.threshold, nms=args.nms, iou_threshold=args.iou_threshold,
            filter_contained=args.filter,
        )

        for frame, image, result in zip(batch, images, results):
            output = out_dir / f"{frame.stem}_pred.png"
            plot_result(image, result).save(output)
            boxes = [
                {"score": round(s.item(), 4), "label": l.item(), "box_xyxy": [round(v, 1) for v in b.tolist()]}
                for s, l, b in zip(result["scores"], result["labels"], result["boxes"])
            ]
            predictions.append({"frame": str(frame), "boxes": boxes})
            print(f"{frame.name}: {len(boxes)} boxes" + (f", max score {max(b['score'] for b in boxes):.2f}" if boxes else ""))

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=1)

    print(f"{len(frames)} frames, saved images and predictions.json to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Input image(s) and/or folder(s) of frames")
    parser.add_argument("checkpoint", type=str, help="Path to the .pth checkpoint (e.g. best.pth)")
    parser.add_argument("-c", "--config", type=str,
                        default=str(RTDETR_DIR / "configs/rtdetrv2/rtdetrv2_r18vd_lumen.yml"),
                        help="Training config the checkpoint was trained with")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum confidence score to keep a box")
    parser.add_argument("--nms", action="store_true", help="Apply non-maximum suppression to remove overlapping boxes")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU above which overlapping boxes are suppressed (only with --nms)")
    parser.add_argument("--filter", action="store_true",
                        help="Filter out contained boxes based on confidence (applied after --threshold and --nms)")
    parser.add_argument("-o", "--output-dir", type=str, default="predictions",
                        help="Folder for <frame name>_pred.png images and predictions.json")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per forward pass")
    args = parser.parse_args()
    main(args)
