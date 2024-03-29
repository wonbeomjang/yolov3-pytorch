import random

import numpy as np
import yaml
import argparse

import torch.optim
import torchvision.transforms.functional as TF
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from models.yolo import YOLOv3, YOLOModelInterface
from loss import YOLOv3Loss
from dataloader import get_loader
from utils import AverageMeter, Logger
from utils.bbox import non_maximum_suppression, get_map, xywh2xyxy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
random.seed(100)


def train_one_epoch(net: YOLOModelInterface, criterion: YOLOv3Loss, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader, lr: float, epoch: int = 0, num_epoch: int = 0):
    net = net.train()
    pbar = tqdm(data_loader, total=len(data_loader))
    device = next(net.parameters()).device
    image = None
    target = None

    loss_avg = AverageMeter()
    loss_coord_avg = AverageMeter()
    loss_conf_avg = AverageMeter()
    loss_cls_avg = AverageMeter()

    anchor = (net.yolo_1.scaled_anchors, net.yolo_2.scaled_anchors, net.yolo_3.scaled_anchors)

    for image, target, path in pbar:
        image = image.to(device)
        target = target.to(device)

        pred = net(image)
        loss, loss_detail = criterion(pred, target, anchor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.data)
        loss_coord_avg.update(loss_detail["coord"])
        loss_conf_avg.update(loss_detail["conf"])
        loss_cls_avg.update(loss_detail["cls"])

        pbar.set_description(f"[{epoch}/{num_epoch}] Loss: {loss_avg.avg:.4f} | Coord: {loss_coord_avg.avg:.4f}, "
                             f"Confidence: {loss_conf_avg.avg:.4f}, Class: {loss_cls_avg.avg:.4f}, Learning Rate: {lr}")

    if image is not None and target is not None:
        net = net.eval()
        image = image[0:1, ...].to(device)
        preds = net(image)
        preds = non_maximum_suppression(preds)
        if preds is not None:
            preds = preds.cpu()
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)[0]
            image = draw_bounding_boxes(TF.resize(image, [416, 416]), xywh2xyxy(preds)[..., 0:4]).float().div(255)
        logger.log_image(image)

    return {"train/loss": loss_avg.avg, "train/loss_coord": loss_coord_avg.avg,
            "train/loss_conf": loss_conf_avg.avg, "train/loss_cls": loss_cls_avg.avg, "train/learning_rate": lr}


def val(net: YOLOModelInterface, criterion: YOLOv3Loss, data_loader: DataLoader, epoch: int = 0,
        num_epoch: int = 0):

    with torch.no_grad():
        net = net.train()
        pbar = tqdm(data_loader, total=len(data_loader))
        device = next(net.parameters()).device
        loss_avg = AverageMeter()
        loss_coord_avg = AverageMeter()
        loss_conf_avg = AverageMeter()
        loss_cls_avg = AverageMeter()
        map_avg = AverageMeter()

        anchor = (net.yolo_1.scaled_anchors, net.yolo_2.scaled_anchors, net.yolo_3.scaled_anchors)
        yolo = (net.yolo_1, net.yolo_2, net.yolo_3)

        for image, target, path in pbar:
            image = image.to(device)
            target = target.to(device)

            pred = net(image)
            loss, loss_detail = criterion(pred, target, anchor)

            loss_avg.update(loss.data)
            loss_coord_avg.update(loss_detail["coord"])
            loss_conf_avg.update(loss_detail["conf"])
            loss_cls_avg.update(loss_detail["cls"])

            batch_size = image.size(0)
            transformed_pred = []
            for p, y in zip(pred, yolo):
                transformed_pred += [y.train2eval_format(p, batch_size)]

            transformed_pred = non_maximum_suppression(transformed_pred)
            if transformed_pred is not None:
                res = get_map(transformed_pred, target)
                map_avg.update(res["map"].item())

            pbar.set_description(f"[{epoch}/{num_epoch}] Loss: {loss_avg.avg:.4f} | Coord: {loss_coord_avg.avg:.4f}, "
                                 f"Confidence: {loss_conf_avg.avg:.4f}, Class: {loss_cls_avg.avg:.4f}, "
                                 f"MAP: {map_avg.avg: .4f} Validation...")

    return {"val/loss": loss_avg.avg, "val/loss_coord": loss_coord_avg.avg,
            "val/loss_conf": loss_conf_avg.avg, "val/loss_cls": loss_cls_avg.avg, "val/map": map_avg.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="data/voc.yaml")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--coord_scale", type=float, default=2.0)
    parser.add_argument("--no_obj_scale", type=float, default=100)
    parser.add_argument("--obj_scale", type=float, default=1)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=416)
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(project="yolov3-pytorch", config=args, resume=args.resume)

    with open(args.data) as f:
        data_file = yaml.load(f, Loader=yaml.FullLoader)

    net = YOLOv3(data_file["nc"], args.image_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    criterion = YOLOv3Loss(args.conf_threshold, args.coord_scale, args.no_obj_scale, args.obj_scale, args.class_scale)

    train_loader, val_loader = get_loader(args.data, args.image_size, args.batch_size, args.scale)

    sample_image = torch.randn((1, 3, args.image_size, args.image_size)).to(device)
    net.forward(sample_image)
    start_epoch = 0

    if args.resume:
        state_dict = logger.load_state_dict(map_location=device)
        net.load_state_dict(state_dict["state_dict"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        start_epoch = state_dict["epoch"]
        logger.metrix = state_dict["metrix"]

    for i in range(start_epoch, args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(net, criterion, optimizer, train_loader, lr, i, args.num_epochs)
        val_loss = val(net, criterion, val_loader, i, args.num_epochs)
        lr_scheduler.step(val_loss["val/loss"])

        logger.log(train_loss)
        logger.log(val_loss)
        logger.save_model(net, optimizer, lr_scheduler, metrix=val_loss["val/loss"], epoch=i)
        logger.end_epoch()

    logger.finish()
