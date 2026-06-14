"""Tests for inference.postprocess.

Runnable either with pytest::

    python -m pytest tests/test_postprocess.py

or directly (no pytest required)::

    python tests/test_postprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.postprocess import (
    _nms_c,
    clip_box,
    iou,
    letterbox_params,
    nms,
    scale_coords_letterbox,
    select_target,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_box_conversions_roundtrip():
    box = (10, 20, 30, 40)  # xywh
    xyxy = xywh_to_xyxy(box)
    assert xyxy == (10.0, 20.0, 40.0, 60.0)
    assert xyxy_to_xywh(xyxy) == (10.0, 20.0, 30.0, 40.0)


def test_clip_box_no_negative_size():
    # Box partly outside, plus inverted corners.
    clipped = clip_box((-5, -5, 50, 50), width=40, height=30)
    x1, y1, x2, y2 = clipped
    assert x1 == 0.0 and y1 == 0.0
    assert x2 == 40.0 and y2 == 30.0
    assert (x2 - x1) >= 0 and (y2 - y1) >= 0

    inverted = clip_box((30, 20, 10, 5), width=100, height=100)
    ix1, iy1, ix2, iy2 = inverted
    assert ix2 >= ix1 and iy2 >= iy1


def test_iou_overlap_cases():
    a = (0, 0, 10, 10)
    # Identical -> 1.0
    assert abs(iou(a, (0, 0, 10, 10)) - 1.0) < 1e-9
    # Disjoint -> 0.0
    assert iou(a, (20, 20, 30, 30)) == 0.0
    # Fully contained: 5x5 inside 10x10 -> 25/100 = 0.25
    assert abs(iou(a, (0, 0, 5, 5)) - 0.25) < 1e-9
    # Half overlap: shift right by 5 -> inter 50, union 150 -> 1/3
    assert abs(iou(a, (5, 0, 15, 10)) - (50.0 / 150.0)) < 1e-9


def test_nms_filters_high_overlap_low_score():
    boxes = [
        (0, 0, 10, 10),     # score 0.9 -> kept
        (1, 1, 11, 11),     # heavy overlap, score 0.8 -> dropped
        (100, 100, 110, 110),  # disjoint, score 0.7 -> kept
    ]
    scores = [0.9, 0.8, 0.7]
    keep = nms(boxes, scores, iou_threshold=0.45)
    assert keep[0] == 0
    assert 2 in keep
    assert 1 not in keep
    assert len(keep) == 2


def test_nms_empty():
    assert nms([], [], 0.5) == []


def test_c_nms_matches_python_when_available():
    boxes = np.array(
        [
            (0, 0, 10, 10),
            (1, 1, 11, 11),
            (100, 100, 110, 110),
            (101, 101, 111, 111),
            (50, 50, 70, 80),
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.75, 0.6], dtype=np.float64)
    keep_c = _nms_c(boxes, scores, 0.45)
    if keep_c is None:
        return

    import inference.postprocess as postprocess

    original = postprocess._nms_c
    postprocess._nms_c = lambda *_args, **_kwargs: None
    try:
        keep_py = postprocess.nms(boxes, scores, 0.45)
    finally:
        postprocess._nms_c = original
    assert keep_c == keep_py


def test_letterbox_params_and_scale_back():
    # 480x640 original -> 640x640 letterbox: ratio 1.0 (640/640) vs 1.333; min=1.0
    orig = (480, 640)
    new = (640, 640)
    ratio, (pad_x, pad_y) = letterbox_params(orig, new)
    assert abs(ratio - 1.0) < 1e-9
    assert abs(pad_x - 0.0) < 1e-9
    assert abs(pad_y - 80.0) < 1e-9  # (640 - 480)/2

    # A box in model space maps back inside the original frame.
    model_box = np.array([[100, 180, 200, 280]], dtype=np.float64)
    back = scale_coords_letterbox(model_box, orig, new)[0]
    # y had 80 px padding removed
    assert abs(back[1] - 100.0) < 1e-6
    assert abs(back[3] - 200.0) < 1e-6
    # all within bounds, no negative size
    assert 0 <= back[0] <= 640 and 0 <= back[2] <= 640
    assert 0 <= back[1] <= 480 and 0 <= back[3] <= 480
    assert back[2] >= back[0] and back[3] >= back[1]


def test_select_target_class_filter_and_ranking():
    dets = [
        {"bbox": (0, 0, 20, 20), "label": "cat", "confidence": 0.9, "class_id": 15},
        {"bbox": (300, 300, 40, 40), "label": "person", "confidence": 0.6, "class_id": 0},
        {"bbox": (310, 230, 80, 80), "label": "person", "confidence": 0.8, "class_id": 0},
    ]
    center = (320, 240)
    chosen = select_target(dets, center, target_class="person")
    assert chosen is not None
    assert chosen["label"] == "person"
    # The larger, more centered person box wins.
    assert chosen["bbox"] == (310, 230, 80, 80)

    # No matching class -> None.
    assert select_target(dets, center, target_class="dog") is None
    # Empty list -> None.
    assert select_target([], center) is None


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
