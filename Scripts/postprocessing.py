import torchvision.ops as ops

def contained_bbox(boxes, scores, margin=5):
    """
    Filters out bounding boxes that are completely contained within other bounding boxes based on confidence scores.
    Iterate over the boxes sorted by area (largest first). For each box, check if it contains any other boxes. For each list of contained boxes,
    check if the confidence score of any of the contained boxes is higher than the current box's score. If so, filter out the big box.
    Otherwise, filter out the contained boxes.

    Parameters:
    boxes: list of [x1, y1, x2, y2] boxes.
    scores: list of confidence scores, one per box.

    Returns:
    List of indices (into boxes) of the boxes to keep, in their original order.
    """

    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    order_indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)

    # indices, not boxes, so identical duplicate boxes are told apart
    removed = set()
    for i in order_indices:
        if i in removed:
            continue

        contained = []
        for j in order_indices:
            if j == i or j in removed:
                continue
            if (boxes[j][0] >= boxes[i][0] - margin and boxes[j][1] >= boxes[i][1] - margin and
                boxes[j][2] <= boxes[i][2] + margin and boxes[j][3] <= boxes[i][3] + margin):
                contained.append(j)

        if len(contained) > 0:
            keep_contained = False
            for c in contained:
                if scores[c] > scores[i]:
                    removed.add(i)
                    keep_contained = True
                    break

            if not keep_contained:
                removed.update(contained)

    return [i for i in range(len(boxes)) if i not in removed]

def postprocessing(predictions, threshold=0.3, nms=False, iou_threshold=0.5, filtering=False):
    """
    Apply confidence threshold, then NMS (if nms is True), then removal of contained boxes with contained_bbox (if filtering is True).

    Parameters:
    predictions: list with one dict per image, as returned by the RT-DETR postprocessor, with tensors under the
        keys "boxes" ([N, 4], x1 y1 x2 y2), "scores" ([N]) and "labels" ([N]).
    threshold: minimum confidence score to keep a box.
    nms: if True, apply NMS.
    iou_threshold: IoU above which the lower-scoring of two overlapping boxes is removed (only used with nms).
    filtering: if True, filter out contained boxes with contained_bbox.

    Returns:
    List with one dict per image, with the same keys as the input and only the boxes that were kept.
    """

    results = []
    for pred in predictions:
        keep = pred["scores"] > threshold
        result = {k: v[keep] for k, v in pred.items()}

        if nms:
            keep = ops.batched_nms(result["boxes"], result["scores"], result["labels"], iou_threshold)
            result = {k: v[keep] for k, v in result.items()}

        if filtering:
            keep = contained_bbox(result["boxes"].tolist(), result["scores"].tolist())
            result = {k: v[keep] for k, v in result.items()}

        results.append(result)

    return results