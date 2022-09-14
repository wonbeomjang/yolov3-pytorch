import pprint
from typing import List, Optional

import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision


def xywh2xyxy(xywh: Tensor, width: int = 1, height: int = 1) -> Tensor:
    xywh[..., 0], xywh[..., 2] = xywh[..., 0] - torch.div(xywh[..., 2], 2.0), xywh[..., 0] + torch.div(xywh[..., 2], 2.0)
    xywh[..., 1], xywh[..., 3] = xywh[..., 1] - torch.div(xywh[..., 3], 2.0), xywh[..., 1] + torch.div(xywh[..., 3], 2.0)

    return xywh


def xyxy2xywh(xyxy: Tensor) -> Tensor:
    xyxy[..., 0], xyxy[..., 2] = torch.div(xyxy[..., 0] + xyxy[..., 2], 2.), xyxy[..., 2] - xyxy[..., 0]
    xyxy[..., 1], xyxy[..., 3] = torch.div(xyxy[..., 1] + xyxy[..., 3], 2.), xyxy[..., 3] - xyxy[..., 1]

    return xyxy


def get_iou(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.max(b1_x2, b2_x2)
    y2 = torch.max(b1_y2, b2_y2)

    intersection = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)
    bbox1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    bbox2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    res = intersection / (bbox1_area + bbox2_area - intersection + 1e-16)
    return res


def get_map(preds, target, image_size=416):
    batch_size = preds.size(0)

    ps = []
    ts = []

    for i in range(batch_size):
        p = preds[preds[:, -1] == i]
        t = target[target[:, 0] == i]
        ps += [dict(boxes=p[:, 0:4], scores=p[:, 4], labels=p[:, 6])]
        ts += [dict(boxes=t[:, 1:5] * image_size, scores=t[:, 4], labels=t[:, 5])]

    metric = MeanAveragePrecision()
    metric.update(ps, ts)

    return metric.compute()


def non_maximum_suppression(pred: List[Tensor], obj_threshold: float = 0.3,
                            nms_threshold: float = 0.5) -> Optional[Tensor]:
    """
    :param pred: list of output of YOLOLayer / [batch_size, num_bbox, 5 + num_classes]
    :param obj_threshold: minimum probability of detect object
    :param nms_threshold: minimum IOU of bbox that is recognized as same object
    :return: output: [xmin, ymin, xmax, ymax, conf, probs, class_index, batch_index]
    """

    pred = torch.cat(pred, dim=1)
    pred[..., :4] = xywh2xyxy(pred[..., :4])
    output = []

    for i, bbox in enumerate(pred):
        bbox = bbox[bbox[..., 4] > obj_threshold]

        if not bbox.size(0):
            continue

        score = bbox[..., 4] * bbox[..., 5:].max(1)[0]
        bbox = bbox[(-score).argsort()]
        class_prob, class_pred = bbox[..., 5:].max(1, keepdim=True)
        detections = torch.cat([bbox[..., :5], class_prob.float(), class_pred.float(), torch.full(class_pred.shape, i)], dim=1)

        boxes = []
        while detections.size(0):
            high_iou = get_iou(detections[0:1, :4], detections[:, :4]) > nms_threshold
            class_match = detections[0, -2] == detections[:, -2]
            suppression_index = high_iou & class_match

            weight = detections[suppression_index, 4:5]
            detections[0, :4] = torch.div((weight * detections[suppression_index, :4]).sum(0), weight.sum()).round()

            boxes += [detections[0]]
            detections = detections[~suppression_index]

        if boxes:
            output += [torch.stack(boxes)]

    if output:
        output = torch.cat(output, dim=0)
        return xyxy2xywh(output)
    return None


def rescale_bbox(bb, W, H):
    x, y, w, h = bb
    return [x * W, y * H, w * W, h * H]


# 송서연