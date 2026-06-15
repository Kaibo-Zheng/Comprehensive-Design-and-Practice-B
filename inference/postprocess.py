"""与推理框架无关的目标检测后处理。

该模块由 YOLO ONNX 与 RKNN 检测器共用，避免两种运行时互相依赖。
模块包含边界框转换、IoU、NMS、等比例填充坐标映射和主目标选择等逻辑。
边界框使用两种表示方式：

- ``xywh``：左上角坐标加宽高，也是跟踪器 Target 使用的格式。
- ``xyxy``：左上角与右下角坐标，便于计算 IoU 和 NMS。

函数可以接收普通元组、列表或 numpy 数组；需要标量时返回普通 Python 数值，
便于后续序列化为 JSON。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Sequence[float]


# --------------------------------------------------------------------------- #
# 边界框格式转换
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(box: Box) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (float(x), float(y), float(x) + float(w), float(y) + float(h))


def xyxy_to_xywh(box: Box) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1))


def clip_box(box_xyxy: Box, width: float, height: float) -> Tuple[float, float, float, float]:
    """将 xyxy 边界框裁剪到图像范围内。

    函数会保证右下角不小于左上角，因此转换后的宽高不会为负数。
    """

    x1, y1, x2, y2 = box_xyxy
    x1 = min(max(0.0, float(x1)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    x2 = min(max(0.0, float(x2)), float(width))
    y2 = min(max(0.0, float(y2)), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def scale_box(box_xyxy: Box, scale_x: float, scale_y: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (float(x1) * scale_x, float(y1) * scale_y, float(x2) * scale_x, float(y2) * scale_y)


# --------------------------------------------------------------------------- #
# 交并比计算
# --------------------------------------------------------------------------- #
def iou(box_a: Box, box_b: Box) -> float:
    """计算两个 xyxy 边界框的 IoU；无重叠或退化框返回 0。"""

    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def iou_batch(boxes_xyxy: np.ndarray, box: Box) -> np.ndarray:
    """计算一组 xyxy 边界框与单个边界框之间的 IoU。"""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    bx1, by1, bx2, by2 = (float(v) for v in box)

    inter_x1 = np.maximum(boxes[:, 0], bx1)
    inter_y1 = np.maximum(boxes[:, 1], by1)
    inter_x2 = np.minimum(boxes[:, 2], bx2)
    inter_y2 = np.minimum(boxes[:, 3], by2)

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0.0, inter / union, 0.0)
    return out.astype(np.float64)


# --------------------------------------------------------------------------- #
# 非极大值抑制
# --------------------------------------------------------------------------- #
def nms(boxes_xyxy: Sequence[Box], scores: Sequence[float], iou_threshold: float = 0.45) -> List[int]:
    """贪心非极大值抑制，按置信度从高到低返回保留索引。"""

    if len(boxes_xyxy) == 0:
        return []

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores_arr.shape[0] != boxes.shape[0]:
        raise ValueError("scores length must match boxes length")

    keep_c = _nms_c(boxes, scores_arr, iou_threshold)
    if keep_c is not None:
        return keep_c

    order = scores_arr.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_batch(boxes[rest], boxes[i])
        order = rest[ious <= iou_threshold]
    return keep


def _nms_c(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> Optional[List[int]]:
    """尝试使用可选的 C 扩展 NMS；不可用时返回 None。"""

    def load_installed_cconv():
        return importlib.import_module("cpp_conv._cconv")

    def load_local_cconv():
        accel_dir = Path(__file__).resolve().parents[1] / "acceleration"
        for pattern in ("_cconv*.so", "_cconv*.pyd"):
            for shared_object in sorted(accel_dir.glob(pattern)):
                spec = importlib.util.spec_from_file_location("_cconv", shared_object)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        for pattern in ("build/**/_cconv*.so", "build/**/_cconv*.pyd"):
            for shared_object in sorted(accel_dir.glob(pattern)):
                spec = importlib.util.spec_from_file_location("_cconv", shared_object)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        raise ImportError(f"local _cconv extension not found under {accel_dir}")

    try:
        _cconv = load_installed_cconv()
    except Exception:
        try:
            _cconv = load_local_cconv()
        except Exception:
            return None
    if getattr(_cconv, "nms_xyxy", None) is None:
        try:
            _cconv = load_local_cconv()
        except Exception:
            return None

    nms_xyxy = getattr(_cconv, "nms_xyxy", None)
    if nms_xyxy is None:
        return None

    keep = nms_xyxy(
        np.ascontiguousarray(boxes, dtype=np.float64),
        np.ascontiguousarray(scores, dtype=np.float64),
        float(iou_threshold),
    )
    return [int(i) for i in np.asarray(keep, dtype=np.intp)]


# --------------------------------------------------------------------------- #
# 等比例填充坐标映射。
# --------------------------------------------------------------------------- #
def letterbox_params(
    orig_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
) -> Tuple[float, Tuple[float, float]]:
    """计算等比例填充的缩放比例以及左侧、上侧填充量。

    ``orig_shape`` 与 ``new_shape`` 均为高度、宽度形式；填充量表示左上侧新增的像素数。
    """

    oh, ow = orig_shape
    nh, nw = new_shape
    ratio = min(nh / oh, nw / ow)
    resized_w = ow * ratio
    resized_h = oh * ratio
    pad_x = (nw - resized_w) / 2.0
    pad_y = (nh - resized_h) / 2.0
    return ratio, (pad_x, pad_y)


def scale_coords_letterbox(
    boxes_xyxy: np.ndarray,
    orig_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
) -> np.ndarray:
    """把模型输入空间中的 xyxy 边界框映射回原图坐标。

    返回值会被裁剪到原图范围内。
    """

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    oh, ow = orig_shape
    ratio, (pad_x, pad_y) = letterbox_params(orig_shape, new_shape)

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, oh)
    return boxes


# --------------------------------------------------------------------------- #
# 主目标选择
# --------------------------------------------------------------------------- #
def select_target(
    detections: List[Dict],
    frame_center: Tuple[float, float],
    target_class: Optional[str] = None,
    center_weight: float = 0.25,
) -> Optional[Dict]:
    """从检测结果列表中选择一个主目标。

    每个检测结果应包含如下字段：

        {"bbox": (x, y, w, h), "label": "person", "confidence": 0.87,
         "class_id": 0}

    如果指定了目标类别，则只考虑类别匹配的候选框。最终按照面积和中心距离综合排序，
    优先选择面积较大且更靠近画面中心的目标。
    """

    if not detections:
        return None

    if target_class is not None:
        candidates = [d for d in detections if d.get("label") == target_class]
    else:
        candidates = list(detections)

    if not candidates:
        return None

    cx0, cy0 = frame_center

    def rank(det: Dict) -> float:
        x, y, w, h = det["bbox"]
        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
        return float(w * h) - center_weight * dist

    return max(candidates, key=rank)


# --------------------------------------------------------------------------- #
# YOLO 预处理与输出解码，供 ONNX 和 RKNN 检测器共用
# --------------------------------------------------------------------------- #
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def letterbox(image, new_size: int = 640, color: int = 114):
    """将 BGR 图像缩放并填充为正方形等比例填充输入。

    函数只返回填充后的图像；坐标还原由 scale_coords_letterbox 根据原图尺寸和模型尺寸计算。
    """

    import cv2  # 局部导入，避免没有摄像头环境时影响模块加载。

    oh, ow = image.shape[:2]
    ratio = min(new_size / oh, new_size / ow)
    nw, nh = int(round(ow * ratio)), int(round(oh * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas


def preprocess_yolo(image, new_size: int = 640) -> np.ndarray:
    """将 BGR uint8 图像转为 NCHW、RGB、float32、[0,1] 的 YOLO 输入。"""

    padded = letterbox(image, new_size)
    rgb = padded[:, :, ::-1]  # BGR 转 RGB。
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw[np.newaxis, ...]  # 增加批量维度。


def decode_yolo_output(
    raw: np.ndarray,
    orig_shape: Tuple[int, int],
    input_size: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    num_classes: int = 80,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """将单张图像的 YOLOv5/YOLOv8/YOLOv11 原始输出解码为检测结果。

    支持 ``(1, C, N)`` 和 ``(1, N, C)`` 两种输出布局。函数会在模型输入空间解码
    ``cx, cy, w, h``，再映射回原图，经过 NMS 后返回 bbox、label、confidence
    和 class_id。
    """

    names = list(class_names) if class_names is not None else COCO_NAMES
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    # 统一整理为 (N, C)，类别和边界框所在轴通常较短。
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (C, N) -> (N, C)

    n_attrs = arr.shape[1]
    if n_attrs == num_classes + 5:
        # YOLOv5：中心点、宽高、目标分数和类别分数。
        boxes_cxcywh = arr[:, :4]
        obj = arr[:, 4:5]
        class_scores = arr[:, 5:] * obj
    elif n_attrs == num_classes + 4:
        # YOLOv8/v11：中心点、宽高和类别分数。
        boxes_cxcywh = arr[:, :4]
        class_scores = arr[:, 4:]
    else:
        raise ValueError(
            f"Unexpected YOLO output: {n_attrs} attrs per anchor "
            f"(expected {num_classes + 4} for v8/v11 or {num_classes + 5} for v5). "
            f"Raw shape was {np.asarray(raw).shape}."
        )

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = confidences >= conf_thres
    if not np.any(keep_mask):
        return []

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    class_ids = class_ids[keep_mask]
    confidences = confidences[keep_mask]

    # 在模型输入空间中将 cxcywh 转为 xyxy。
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 映射回原始图像坐标。
    xyxy = scale_coords_letterbox(xyxy, orig_shape, (input_size, input_size))

    keep = nms(xyxy, confidences, iou_threshold=iou_thres)

    detections: List[Dict] = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        cid = int(class_ids[i])
        label = names[cid] if 0 <= cid < len(names) else str(cid)
        detections.append({
            "bbox": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            "label": label,
            "confidence": float(confidences[i]),
            "class_id": cid,
        })
    return detections
