from typing import List, Tuple, Optional
from abc import *

import torch
import torch.nn as nn
from torch import Tensor

from models.backbone import DarkNet53
from models.block import ConvBatchReLU
from torch.quantization import fuse_modules


class YOLOModelInterface(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(YOLOModelInterface, self).__init__()
        self.yolo_1: Optional[YOLOLayer] = None
        self.yolo_2: Optional[YOLOLayer] = None
        self.yolo_3: Optional[YOLOLayer] = None

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pass


class FPNDownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: nn.Module = nn.LeakyReLU):
        super(FPNDownSample, self).__init__()

        self.conv = nn.Sequential(
            ConvBatchReLU(in_channels, out_channels, 1, stride=1, padding=0, relu=relu),
            ConvBatchReLU(out_channels, out_channels * 2, 3, stride=1, padding=1, relu=relu),
            ConvBatchReLU(out_channels * 2, out_channels, 1, stride=1, padding=0, relu=relu),
            ConvBatchReLU(out_channels, out_channels * 2, 3, stride=1, padding=1, relu=relu),
            ConvBatchReLU(out_channels * 2, out_channels, 1, stride=1, padding=0, relu=relu)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class YOLOLayer(nn.Module):
    def __init__(self, in_channels: int, anchors: List[Tuple[int, int]], num_classes: int, image_size: int = 416,
                 relu: nn.Module = nn.LeakyReLU):
        super(YOLOLayer, self).__init__()
        self.scaled_anchors = None
        self.anchor_h = None
        self.anchor_w = None
        self.grid_y = None
        self.grid_x = None
        self.stride = None

        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.anchors = anchors
        self.image_size = image_size
        self.grid_size = 0

        self.adapt_conv = nn.Sequential(
            ConvBatchReLU(in_channels, in_channels * 2, relu=relu),
            nn.Conv2d(in_channels * 2, self.num_anchors * (5 + self.num_classes), 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.adapt_conv(x)

        batch_size = x.size(0)
        grid_size = x.size(2)
        device = x.device

        x = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.contiguous()

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, device)

        if not self.training:
            return self.train2eval_format(x, batch_size)

        return x

    def train2eval_format(self, x, batch_size):
        obj_score = torch.sigmoid(x[..., 4])
        cls_score = torch.sigmoid(x[..., 5:])

        pred_bboxes = self.transform_output(x)
        output = torch.cat((
            pred_bboxes.view(batch_size, -1, 4),
            obj_score.view(batch_size, -1, 1),
            cls_score.view(batch_size, -1, self.num_classes)
        ), dim=-1)
        return output

    def compute_grid_offsets(self, grid_size: int, device: torch.device) -> None:
        self.grid_size = grid_size
        self.stride = self.image_size // self.grid_size

        self.grid_x = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).type(torch.float32)
        self.grid_y = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).type(torch.float32)

        self.scaled_anchors = torch.tensor([(width / self.stride, height / self.stride) for width, height in self.anchors], device=device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def transform_output(self, prediction: Tensor) -> Tensor:
        device = prediction.device
        # B, A, W, H, 5 + C
        x = prediction[..., 0].sigmoid()
        y = prediction[..., 1].sigmoid()
        w = prediction[..., 2].exp()
        h = prediction[..., 3].exp()

        predict_bboxes = torch.zeros_like(prediction[..., :4], device=device)
        predict_bboxes[..., 0] = x.data + self.grid_x
        predict_bboxes[..., 1] = y.data + self.grid_y
        predict_bboxes[..., 2] = w.data * self.anchor_w
        predict_bboxes[..., 3] = h.data * self.anchor_h

        return predict_bboxes * self.stride


class YOLOv3(YOLOModelInterface):
    def __init__(self, num_classes: int, image_size: int = 416, in_channels: int = 3, relu: nn.Module = nn.LeakyReLU):
        super(YOLOv3, self).__init__()
        backbone = DarkNet53(in_channels=in_channels, relu=relu)
        anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]

        self.adj = backbone.adj
        self.block1 = backbone.block1
        self.block2 = backbone.block2
        self.block3 = backbone.block3
        self.block4 = backbone.block4
        self.block5 = backbone.block5

        self.fpn_down_1 = FPNDownSample(1024, 512, relu=relu)
        self.fpn_down_2 = FPNDownSample(512 + 256, 256, relu=relu)
        self.fpn_down_3 = FPNDownSample(256 + 128, 128, relu=relu)

        self.literal_1 = ConvBatchReLU(512, 256, 1, 1, 0, relu=relu)
        self.literal_2 = ConvBatchReLU(256, 128, 1, 1, 0, relu=relu)

        self.yolo_1 = YOLOLayer(512, anchors[2], num_classes, image_size, relu=relu)
        self.yolo_2 = YOLOLayer(256, anchors[1], num_classes, image_size, relu=relu)
        self.yolo_3 = YOLOLayer(128, anchors[0], num_classes, image_size, relu=relu)

        self.upsample = nn.Upsample(scale_factor=2)

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.adj(x)
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        p5 = self.fpn_down_1(c5)
        p4 = self.fpn_down_2(torch.cat((self.upsample(p5), self.literal_1(c4)), dim=1))
        p3 = self.fpn_down_3(torch.cat((self.upsample(p4), self.literal_2(c3)), dim=1))

        yolo_1 = self.yolo_1(p5)
        yolo_2 = self.yolo_2(p4)
        yolo_3 = self.yolo_3(p3)

        return yolo_1, yolo_2, yolo_3

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self._forward_impl(x)


class QuantizableYOLOv3(YOLOv3):
    def __init__(self, num_classes: int, image_size: int = 416, in_channels: int = 3, relu: nn.Module = nn.LeakyReLU):
        super(QuantizableYOLOv3, self).__init__(num_classes, image_size, in_channels, relu)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvBatchReLU):
                fuse_modules(m, ["0", "1", "2"], inplace=True)


if __name__ == "__main__":
    net = QuantizableYOLOv3(10, relu=nn.ReLU)
    net.eval()
    net.fuse_model()

    print(net)
