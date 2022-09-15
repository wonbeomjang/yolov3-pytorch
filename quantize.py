import argparse

import torch
import yaml
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataloader import get_loader
from loss import YOLOv3Loss
from models.yolo import YOLOv3
from train import train_one_epoch, val
from utils import Logger


def quantize_model(net: nn.Module, criterion: nn.Module, optimier: optim.Optimizer, data_loader: DataLoader):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="datasets/custom.yaml")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--coord_scale", type=float, default=2.0)
    parser.add_argument("--no_obj_scale", type=float, default=0.5)
    parser.add_argument("--obj_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=416)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(project="yolov3-pytorch", config=args)

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