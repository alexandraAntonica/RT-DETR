
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


# Each scenario: (description, boxes, ids expected to survive filter_contained_bbox)
# Box: (id, [x1, y1, x2, y2], score)
SCENARIOS = [
    ("big box wins: avg(B, C) = 0.375 < 0.92 -> drop B, C",
     [("A", [40, 40, 460, 460], 0.92),
      ("B", [90, 110, 220, 240], 0.40),
      ("C", [260, 280, 420, 420], 0.35)],
     ["A"]),
    ("contained boxes win: avg(E, F) = 0.825 >= 0.30 -> drop D",
     [("D", [540, 40, 990, 460], 0.30),
      ("E", [590, 110, 760, 270], 0.85),
      ("F", [790, 290, 950, 430], 0.80)],
     ["E", "F"]),
    ("nested: G drops (avg(H, I) = 0.70 >= 0.60), then H drops (0.90 >= 0.50)",
     [("G", [40, 540, 500, 990], 0.60),
      ("H", [90, 610, 380, 900], 0.50),
      ("I", [150, 680, 290, 820], 0.90)],
     ["I"]),
    ("partial overlap, no containment -> keep both",
     [("J", [540, 540, 800, 800], 0.75),
      ("K", [700, 700, 960, 960], 0.65)],
     ["J", "K"]),
]

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


# Box id -> color of its scenario, so redrawn boxes keep their original color
ID_COLORS = {box_id: color for (_, scenario_boxes, _), color in zip(SCENARIOS, COLORS)
             for box_id, _, _ in scenario_boxes}


def draw_boxes(boxes, img_path):
    """Draw boxes (dicts with "box_xyxy", "score" and optionally "id") on a white 1024x1024 image."""
    img = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)

    for box in boxes:
        xyxy = box["box_xyxy"]
        box_id = box.get("id", "")
        color = ID_COLORS.get(box_id, "#d62728")
        draw.rectangle(xyxy, outline=color, width=4)
        draw.text((xyxy[0] + 6, xyxy[1] + 4), f"{box_id} {box['score']:.2f}".strip(), fill=color, font=font)

    img.save(img_path)


def main():
    parser = argparse.ArgumentParser(description="Draw a toy image with overlapping/contained boxes.")
    parser.add_argument("--out_dir", default="toy_example")
    parser.add_argument("--from_json", default=None,
                        help="Draw the boxes of an existing predictions JSON (e.g. the filtered one) instead")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, "r") as f:
            predictions = json.load(f)
        img_path = os.path.splitext(args.from_json)[0] + ".png"
        draw_boxes(predictions[0]["boxes"], img_path)
        print(f"Wrote {img_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    boxes = []
    for desc, scenario_boxes, expected in SCENARIOS:
        for box_id, xyxy, score in scenario_boxes:
            # Same format as the inference output; "id" is extra and only for reference
            boxes.append({"id": box_id, "score": score, "label": 0, "box_xyxy": xyxy})
        print(f"[{', '.join(expected)}]  <- {desc}")

    img_path = os.path.join(args.out_dir, "toy_boxes.png")
    json_path = os.path.join(args.out_dir, "toy_boxes.json")
    draw_boxes(boxes, img_path)
    with open(json_path, "w") as f:
        json.dump([{"frame": os.path.abspath(img_path), "boxes": boxes}], f, indent=2)

    all_expected = sorted(i for _, _, expected in SCENARIOS for i in expected)
    print(f"\nExpected surviving ids: {all_expected}")
    print(f"Wrote {img_path} and {json_path}")


if __name__ == "__main__":
    main()
