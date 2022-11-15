from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class YOLOv3Loss(nn.Module):
    def __init__(self, conf_threshold: float, coord_scale: float, no_obj_scale: float, obj_scale: float,
                 class_scale: float):
        super(YOLOv3Loss, self).__init__()
        self.conf_threshold = conf_threshold
        self.coord_scale = coord_scale
        self.obj_scale = obj_scale
        self.no_obj_scale = no_obj_scale
        self.class_scale = class_scale

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, preds: List[Tensor], target: Tensor, anchors: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        loss = torch.tensor(0.0, device=preds[0].device)
        loss_coord_sum = 0.0
        loss_conf_sum = 0.0
        loss_cls_sum = 0.0

        for i, pred in enumerate(preds):
            x, y, w, h = self.get_yolo_preds(pred)
            yolo_targets = self.get_yolo_targets(pred, anchors[i], target)

            obj_mask = yolo_targets["obj_mask"]
            no_obj_mask = yolo_targets["no_obj_mask"]
            tx = yolo_targets["tx"]
            ty = yolo_targets["ty"]
            tw = yolo_targets["tw"]
            th = yolo_targets["th"]
            tcls = yolo_targets["tcls"]
            t_conf = yolo_targets["t_conf"]
            pred_conf = pred[:, :, :, :, 4]

            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask].sqrt(), tw[obj_mask].sqrt())
            loss_h = self.mse_loss(h[obj_mask].sqrt(), th[obj_mask].sqrt())

            loss_coord = loss_x + loss_y + loss_w + loss_h
            loss_coord *= self.coord_scale

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
            loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], t_conf[no_obj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj

            loss_cls = self.bce_loss(pred[:, :, :, :, 5:][obj_mask], tcls[obj_mask]) * self.class_scale

            loss += loss_coord + loss_conf + loss_cls
            loss_coord_sum += loss_coord.item()
            loss_conf_sum += loss_conf.item()
            loss_cls_sum += loss_cls.item()

        return loss, {"coord": loss_coord_sum, "conf": loss_conf_sum, "cls": loss_cls_sum}

    def get_yolo_preds(self, preds: Tensor):
        x = preds[:, :, :, :, 0]
        y = preds[:, :, :, :, 1]
        w = preds[:, :, :, :, 2]
        h = preds[:, :, :, :, 3]

        x = x.sigmoid()
        y = y.sigmoid()
        w = w.exp()
        h = h.exp()

        return x, y, w, h

    def get_yolo_targets(self, preds: Tensor, anchors: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        device = preds.device
        preds_class = preds[:, :, :, :, 5:]

        batch_size, num_anchors, grid_size, _, num_classes = preds_class.shape

        size_t = batch_size, num_anchors, grid_size, grid_size
        obj_mask = torch.zeros(size_t, device=device, dtype=torch.bool)
        no_obj_mask = torch.ones(size_t, device=device, dtype=torch.bool)
        tx = torch.zeros(size_t, device=device, dtype=torch.float32)
        ty = torch.zeros(size_t, device=device, dtype=torch.float32)
        tw = torch.zeros(size_t, device=device, dtype=torch.float32)
        th = torch.zeros(size_t, device=device, dtype=torch.float32)

        size_t = batch_size, num_anchors, grid_size, grid_size, num_classes
        tcls = torch.zeros(size_t, device=device, dtype=torch.float32)

        t_xy = targets[:, 1:3] * grid_size
        t_wh = targets[:, 3:5] * grid_size
        t_x, t_y = t_xy.t()
        t_w, t_h = t_wh.t()

        grid_i, grid_j = t_xy.long().t()
        # ....? 뭐지
        iou = torch.stack([self.get_iou_wh(anchor, t_wh) for anchor in anchors])
        best_iou, best_anchor_index = iou.max(0)

        batch_index, targets_label = targets[:, 0].long(), targets[:, 5].long()
        obj_mask[batch_index, best_anchor_index, grid_j, grid_i] = True
        no_obj_mask[batch_index, best_anchor_index, grid_j, grid_i] = False

        for i, iou_val in enumerate(iou.t()):
            no_obj_mask[batch_index[i], iou_val > self.conf_threshold, grid_j[i], grid_i[i]] = True

        tx[batch_index, best_anchor_index, grid_j, grid_i] = t_x - t_x.floor()
        ty[batch_index, best_anchor_index, grid_j, grid_i] = t_y - t_y.floor()

        anchor_w = anchors[best_anchor_index][:, 0]
        anchor_h = anchors[best_anchor_index][:, 1]
        tw[batch_index, best_anchor_index, grid_j, grid_i] = t_w / anchor_w
        th[batch_index, best_anchor_index, grid_j, grid_i] = t_h / anchor_h

        tcls[batch_index, best_anchor_index, grid_j, grid_i, targets_label] = 1

        output = {
            "obj_mask": obj_mask,
            "no_obj_mask": no_obj_mask,
            "tx": tx,
            "ty": ty,
            "tw": tw,
            "th": th,
            "tcls": tcls,
            "t_conf": obj_mask.float(),
        }

        return output

    def get_iou_wh(self, wh1, wh2):
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area