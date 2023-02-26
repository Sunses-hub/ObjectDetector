
from torch import nn
import torch.nn.functional as F
import torch

# Building

class BottleNeck(nn.Module):

    def __init__(self, in_channels=64, out_channels=256, width=64,
                 stride=1, padding=1, residual=True):
        super().__init__()

        # layers
        self.conv1 = nn.Conv2d(in_channels, width, 1, stride)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, padding)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # save the input for residual connection
        identity = x
        # forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.residual:
            if self.in_channels == self.out_channels:
                x += identity
            else:
                for i in range(4):
                    x[:, self.in_channels*i:(self.in_channels*(i+1)), :, :] += identity

        return F.relu(x)

class ResNet50(nn.Module):

    def __init__(self, numbBlocks=[3, 4, 6, 3], in_channel=64):
        super(ResNet50, self).__init__()

        self.in_channel = in_channel

        #self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.res_layers = nn.ModuleList()

        for j, blocks in enumerate(numbBlocks):
            width = self.in_channel * 2 ** j
            for i in range(blocks):
                if i == 0:
                    self.res_layers.append(BottleNeck(width, width*4, width))
                else:
                    self.res_layers.append(BottleNeck(width * 4, width * 4, width))
            # add downsampler
            downsampler = nn.Sequential(nn.Conv2d(width * 4, width * 2, kernel_size=1, stride=2),
                                        nn.BatchNorm2d(width * 2))
            self.res_layers.append(downsampler)

        self.fc = nn.Sequential(nn.Linear(1024, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 3))

    def forward(self, x):
        #x = self.relu(self.bn1(self.conv1(x)))
        #x = self.maxpool(x)
        for layer in self.res_layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        return output

class BeastNet(nn.Module):

    def __init__(self):
        super(BeastNet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))

        #self.conv = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
        #                          nn.ReLU(inplace=True)) # can be changed
        # image classifier
        self.resnet = ResNet50()
        # regression (BB detector)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(64, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        h = self.conv(x)
        print(h.shape)
        h2 = h.clone()
        # output of classifier
        y1 = self.resnet(h)
        # output of regressor
        y2 = self.conv_seq(h2)
        y2 = torch.flatten(y2, 1)
        y2 = self.fc(y2)
        return y1, y2


if __name__ == "__main__":
    """
    # test the Bottleneck block
    x = torch.randn((4,3,256,256))
    block1 = BottleNeck(in_channels=3, out_channels=3, width=3, downsample=False)
    y = block1(x)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    

    x = torch.randn((4, 3, 224, 224))
    resnet = ResNet50()
    y = resnet(x)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print("Parameters:", len(list(resnet.parameters())))
    """

    x2 = torch.randn((4, 3, 256, 256))
    beast = BeastNet()
    y1, y2 = beast(x2)
    print("y1 shape:", y1.shape)
    print("y2 shape:", y2.shape)

