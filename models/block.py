import torch.nn as nn


def conv_batch(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
               relu: nn.Module = nn.LeakyReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        relu())
