import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as F
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes

from dataloader import get_loader
from model.yolov3 import YOLOv3
from utils.bbox import non_maximum_suppression, xywh2xyxy


def xywh2xyminmax(boxes):
    x, y, w, h = boxes
    return int(x - w / 2), int(x + w / 2), int(y - h / 2), int(y + h / 2)


def get_bbox(net: nn.Module, image: torch.Tensor, conf_threshold: float = 0.5):
    preds, anchors = net(image)
    bbox = {"class": [], "boxes": []}

    for pred, anchor in zip(preds, anchors):
        pred = pred.squeeze()

        for pred_bbox in pred:
            if pred_bbox[4] < conf_threshold:
                continue
            cls = pred_bbox[5:].argmax(0)

            bbox["boxes"] += [torch.tensor(xywh2xyminmax(pred_bbox[:4]))]
            bbox["class"] += [f"{cls}"]
    bbox["boxes"] = torch.stack(bbox["boxes"])
    return bbox


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def load_image(path: str):
    image = Image.open(path)
    x = F.to_tensor(image)
    x = x.unsqueeze(0)
    return x


if __name__ == "__main__":
    import torchvision.transforms.functional  as TF
    num_classes = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = YOLOv3(num_classes).to(device)
    train_loader, val_loader = get_loader("data/custom.yaml", 416, 1, 1)
    images, target, path = next(iter(train_loader))
    net.load_state_dict(torch.load("checkpoints/best.pt", map_location=device)["state_dict"])

    image = read_image(path[0])
    channel, width, height = image.shape
    labels = list(map(str, target[..., -1].tolist()))
    bbox = target[..., 1:5]
    bbox = xywh2xyxy(bbox, width, height)
    # out_image = draw_bounding_boxes(image, bbox, labels)
    # plt.imshow(TF.to_pil_image(out_image))
    # plt.show()

    images = images.to(device)
    preds, anchor = net(images)
    preds = non_maximum_suppression(preds)[0].cpu()
    out_image = draw_bounding_boxes(image, preds[..., :4])
    plt.imshow(TF.to_pil_image(out_image))
    plt.show()
    #
    #
