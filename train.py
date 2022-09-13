import os
import random

import numpy as np
import yaml
import argparse

import torch.optim
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models.yolo import YOLOv3, YOLOModelInterface
from loss import YOLOv3Loss
from dataloader import get_loader
from utils import AverageMeter, Logger

torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(100)
random.seed(100)


def train_one_epoch(net: YOLOModelInterface, criterion: YOLOv3Loss, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader, lr: float, epoch: int = 0, num_epoch: int = 0):
    net = net.train()
    pbar = tqdm(data_loader, total=len(data_loader))
    device = next(net.parameters()).device
    loss_avg = AverageMeter()
    loss_coord_avg = AverageMeter()
    loss_conf_avg = AverageMeter()
    loss_cls_avg = AverageMeter()

    anchor = (net.yolo_1.scaled_anchors, net.yolo_2.scaled_anchors, net.yolo_3.scaled_anchors)

    for image, target, _ in pbar:
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

        anchor = (net.yolo_1.scaled_anchors, net.yolo_2.scaled_anchors, net.yolo_3.scaled_anchors)

        for image, target, _ in pbar:
            image = image.to(device)
            target = target.to(device)

            pred = net(image)
            loss, loss_detail = criterion(pred, target, anchor)

            loss_avg.update(loss.data)
            loss_coord_avg.update(loss_detail["coord"])
            loss_conf_avg.update(loss_detail["conf"])
            loss_cls_avg.update(loss_detail["cls"])

            pbar.set_description(f"[{epoch}/{num_epoch}] Loss: {loss_avg.avg:.4f} | Coord: {loss_coord_avg.avg:.4f}, "
                                 f"Confidence: {loss_conf_avg.avg:.4f}, Class: {loss_cls_avg.avg:.4f}, Validation...       ")

    return {"val/loss": loss_avg.avg, "val/loss_coord": loss_coord_avg.avg,
            "val/loss_conf": loss_conf_avg.avg, "val/loss_cls": loss_cls_avg.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="datasets/custom.yaml")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--coord_scale", type=float, default=1.0)
    parser.add_argument("--no_obj_scale", type=float, default=5.0)
    parser.add_argument("--obj_scale", type=float, default=5.0)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=416)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(project="kilter-gallery", config=args)

    with open(args.data) as f:
        data_file = yaml.load(f, Loader=yaml.FullLoader)

    net = YOLOv3(data_file["nc"], args.image_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    criterion = YOLOv3Loss(args.conf_threshold, args.coord_scale, args.no_obj_scale, args.obj_scale, args.class_scale)

    train_loader, val_loader = get_loader(args.data, args.image_size, args.batch_size, args.scale)

    sample_image = torch.randn((1, 3, args.image_size, args.image_size)).to(device)
    net.forward(sample_image)

    for i in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(net, criterion, optimizer, train_loader, lr, i, args.num_epochs)
        val_loss = val(net, criterion, val_loader, i, args.num_epochs)
        lr_scheduler.step(val_loss["val/loss"])

        logger.log(train_loss)
        logger.log(val_loss)
        logger.save_model(net, optimizer, lr_scheduler, metrix=val_loss["val/loss"])
        logger.end_epoch()

    logger.finish()
