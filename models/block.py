import torch.nn as nn


class ConvBatchReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 relu: nn.Module = nn.LeakyReLU):
        super(ConvBatchReLU, self).__init__()
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            relu()
        ]
        for idx, module in enumerate(block):
            self.add_module(str(idx), module)
