import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.utils import draw_bounding_boxes

from models.yolo import YOLOv3
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


def show(image):
    if not isinstance(image, list):
        image = [image]
    fix, axs = plt.subplots(ncols=len(image), squeeze=False)
    for i, img in enumerate(image):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def read_image(path: str):
    image = Image.open(path)
    x = F.to_tensor(image)
    x = x.unsqueeze(0)
    return x


if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    from torchvision.utils import save_image
    from dataloader import get_loader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    args = parser.parse_args()

    num_classes = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = YOLOv3(num_classes).to(device)
    train_loader, val_loader = get_loader("EgoHands/data.yaml", 416, 1, 1)
    images, target, path = next(iter(train_loader))
    images, target, path = next(iter(train_loader))
    net.load_state_dict(torch.load("checkpoints/exp1/best.pt", map_location="cpu")["state_dict"])
    net.eval()

    image = read_image(path[0])
    images = images.to(device)
    preds = net(images)
    preds = non_maximum_suppression(preds).cpu()
    out_image = draw_bounding_boxes(TF.resize(image, [416, 416]), xywh2xyxy(target)[..., 1:5] * 416)
    _, _, w, h = images[0:1].shape
    save_image(out_image.float().div(255), "tse.png")
