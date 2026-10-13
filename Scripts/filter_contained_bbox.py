
import json
import os


def contained_bbox_keep(boxes, scores):
    """
    Filters out bounding boxes that are completely contained within other bounding boxes based on confidence scores.
    Iterate over the boxes sorted by area (largest first). For each box, check if it contains any other boxes. For each list of contained boxes,
    check if the average confidence score of the contained boxes is lower than the current box's score. If so, filter out the contained boxes.
    Otherwise, filter out the current box.

    Parameters:
    boxes: list of [x1, y1, x2, y2] boxes.
    scores: list of confidence scores, one per box.

    Returns:
    List of indices (into boxes) of the boxes to keep, in their original order.
    """

    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)

    # indices, not boxes, so identical duplicate boxes are told apart
    removed = set()
    for i in order:
        if i in removed:
            continue

        contained = []
        for j in order:
            if j == i or j in removed:
                continue
            if (boxes[j][0] >= boxes[i][0] and boxes[j][1] >= boxes[i][1] and
                boxes[j][2] <= boxes[i][2] and boxes[j][3] <= boxes[i][3]):
                contained.append(j)

        if len(contained) > 0:
            avg_contained_score = sum([scores[j] for j in contained]) / len(contained)
            if avg_contained_score < scores[i]:
                removed.update(contained)
            else:
                removed.add(i)

    return [i for i in range(len(boxes)) if i not in removed]


def filter_contained_bbox(predictions_file, output_file=None):
    """
    Applies contained_bbox_keep to every frame of a predictions JSON file.

    Parameters:
    predictions_file: Path to the JSON file containing the predictions. It is not modified.
    output_file: Path to write the filtered predictions to.
        Defaults to "<predictions_file without .json>_filtered.json".

    Returns:
    Path to the new json file containing the filtered bounding boxes under the key "boxes".
    """

    if output_file is None:
        output_file = os.path.splitext(predictions_file)[0] + "_filtered.json"

    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    for prediction in predictions:
        bboxes = prediction["boxes"]
        keep = contained_bbox_keep([bbox["box_xyxy"] for bbox in bboxes], [bbox["score"] for bbox in bboxes])
        prediction["boxes"] = [bboxes[i] for i in keep]

    with open(output_file, "w") as f:
        json.dump(predictions, f)

    return output_file


if __name__ == "__main__":
    toy_dir = "/home/alexandra/tunnel-travelling-experiments/RT-DETR/toy_example"
    filter_contained_bbox(f"{toy_dir}/toy_boxes.json", f"{toy_dir}/toy_boxes_filtered.json")
