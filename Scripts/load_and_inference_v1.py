import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageFont, ImageDraw
import torchvision.ops as ops
import argparse
import sys

def load_v1(checkpoint, device):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()

    return image_processor, model

def detect_boxes(image, image_processor, model, device, threshold=0.3, nms=False, iou_threshold=0.5):
    inputs = image_processor(images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]) # use to rescale the bounding boxes to the original image size
    result = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    if nms:
        keep = ops.batched_nms(result["boxes"], result["scores"], result["labels"], iou_threshold)
        result = {k: v[keep] for k, v in result.items()}

    return result

def plot_result(image, result, model):
    image_with_boxes = image.convert("RGB")
    draw = ImageDraw.Draw(image_with_boxes)

    try:
        font = ImageFont.load_default(size=max(12, image_with_boxes.width // 50))
    except TypeError:  # Pillow < 10.1 doesn't accept a size
        font = ImageFont.load_default()

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        x, y, x2, y2 = box.tolist()
        draw.rectangle((x, y, x2, y2), outline="red", width=2)

        text = f"{model.config.id2label[label.item()]} {score.item():.2f}"
        text_box = draw.textbbox((x, y), text, font=font)
        text_height = text_box[3] - text_box[1]
        text_y = y - text_height - 4 if y - text_height - 4 >= 0 else y
        text_box = draw.textbbox((x, text_y), text, font=font)
        draw.rectangle((text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2), fill="red")
        draw.text((x, text_y), text, fill="white", font=font)

    return image_with_boxes

def main(args):

    image_path = args.image
    checkpoint = args.checkpoint
    image = Image.open(image_path).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor, model = load_v1(checkpoint, device)
    result = detect_boxes(
        image, image_processor, model, device,
        threshold=args.threshold, nms=args.nms, iou_threshold=args.iou_threshold,
    )
    image_with_boxes = plot_result(image, result, model)
    image_with_boxes.show()


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    if not sys_args:
        sys_args = [
            "C:\\Users\\AlexandraAntonica\\Documents\\postprocessed_data\\MC_035_clean\\left_lower_lobe_route\\MC_035__left_lower_lobe_route__depth_capture000318_synthesized_image.png",
            "C:\\Users\\AlexandraAntonica\\tunnel-travelling-experiments\\notebooks\\rtdetr-r18-finetune\\checkpoint-3440",
            "--nms",
            "--iou-threshold=0.5"
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum confidence score to keep a box")
    parser.add_argument("--nms", action="store_true", help="Apply non-maximum suppression to remove overlapping boxes")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU above which overlapping boxes are suppressed (only with --nms)")
    args = parser.parse_args(sys_args)
    main(args)