import torch
from torch import nn
from torch.quantization import fuse_modules

from models.block import conv_batch

pretrained_url = "https://github.com/wonbeomjang/parameters/releases/download/parameter/darknet53.pth"


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels, relu: nn.Module = nn.LeakyReLU):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0, relu=relu)
        self.layer2 = conv_batch(reduced_channels, in_channels, relu=relu)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

    def fuse_model(self):
        fuse_modules(self.layer1, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer2, ["0", "1", "2"], inplace=True)


class DarkNet53(nn.Module):
    def __init__(self, in_channels: int = 3, relu: nn.Module = nn.LeakyReLU):
        super(DarkNet53, self).__init__()

        self.adj = conv_batch(in_channels, 32, relu=relu)
        self.block1 = nn.Sequential(conv_batch(32, 64, stride=2, relu=relu),
                                    self.make_layer(in_channels=64, num_blocks=1, relu=relu))
        self.block2 = nn.Sequential(conv_batch(64, 128, stride=2, relu=relu),
                                    self.make_layer(in_channels=128, num_blocks=2, relu=relu))
        self.block3 = nn.Sequential(conv_batch(128, 256, stride=2, relu=relu),
                                    self.make_layer(in_channels=256, num_blocks=8, relu=relu))
        self.block4 = nn.Sequential(conv_batch(256, 512, stride=2, relu=relu),
                                    self.make_layer(in_channels=512, num_blocks=8, relu=relu))
        self.block5 = nn.Sequential(conv_batch(512, 1024, stride=2, relu=relu),
                                    self.make_layer(in_channels=1024, num_blocks=4, relu=relu))

        self.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url))

    def make_layer(self, in_channels, num_blocks, relu: nn.Module = nn.LeakyReLU):
        layers = []
        for i in range(0, num_blocks):
            layers.append(DarkResidualBlock(in_channels, relu=relu))
        return nn.Sequential(*layers)

    def fuse_model(self):
        fuse_modules(self.adj, ["0", "1", "2"], inplace=True)
        for m in self.modules():
            if isinstance(m, DarkResidualBlock):
                m.fuse_model()
        for m in (self.block1, self.block2, self.block3, self.block4, self.block5):
            fuse_modules(m, ["0.0", "0.1", "0.2"], inplace=True)


if __name__ == "__main__":
    net = DarkNet53(3, nn.ReLU)
    net.eval()
    net.fuse_model()
    print(net)