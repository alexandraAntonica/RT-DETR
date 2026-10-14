import torch
from postprocessing import contained_bbox, postprocessing

def make_prediction(boxes, scores):
    return [{"boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.zeros(len(scores), dtype=torch.int64)}]


def test_low_confidence_removal():
    pred = make_prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.1])
    out = postprocessing(pred, threshold=0.45)[0]
    assert out["boxes"].tolist() == [[0, 0, 10, 10]]

def test_nms():
    pred = make_prediction([[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8])  # IoU 0.68
    out = postprocessing(pred, nms=True, iou_threshold=0.3)[0]
    assert out["boxes"].tolist() == [[0, 0, 10, 10]]

def test_contained_bbox_big_box_is_more_confident():
    pred = make_prediction([[0, 0, 100, 100], [10, 10, 20, 20]], [0.9, 0.5])
    out = postprocessing(pred, filtering=True)[0]
    assert out["boxes"].tolist() == [[0, 0, 100, 100]]

def test_contained_bbox_big_box_is_less_confident():
    pred = make_prediction([[0, 0, 100, 100], [10, 10, 20, 20]], [0.6, 0.9])
    out = postprocessing(pred, filtering=True)[0]
    assert out["boxes"].tolist() == [[10, 10, 20, 20]]

def test_contained_bbox_three_contained_boxes_with_high_confidence():
    pred = make_prediction([[0, 0, 100, 100], [10, 10, 80, 80], [15, 15, 60, 60], [20, 20, 70, 70]], [0.6, 0.9, 0.5, 0.8])
    out = postprocessing(pred, filtering=True)[0]
    assert out["boxes"].tolist() == [[10, 10, 80, 80]]

def test_contained_bbox_within_margin_with_high_confidence():
    pred = make_prediction([[0, 0, 100, 100], [10, 10, 104, 104]], [0.6, 0.9])
    out = postprocessing(pred, filtering=True)[0]
    assert out["boxes"].tolist() == [[10, 10, 104, 104]]

def test_all_filters():
    pred = make_prediction(
    [[0, 0, 100, 100],   # big box, score 0.95
        [10, 10, 20, 20],   # inside big box, weaker -> removed by contained filter
        [11, 11, 21, 21],   # duplicate of the one above -> removed by NMS first
        [200, 200, 250, 250]],  # separate, but low score -> removed by threshold
    [0.95, 0.7, 0.6, 0.1],
    )
    out = postprocessing(pred, threshold=0.45, nms=True, iou_threshold=0.3, filtering=True)[0]
    assert out["boxes"].tolist() == [[0, 0, 100, 100]]