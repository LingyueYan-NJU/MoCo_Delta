import torch
import torch as jt
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

        self.bn32_1 = torch.nn.BatchNorm2d(num_features=32)
        self.bn32_2 = torch.nn.BatchNorm2d(num_features=32)
        self.bn32_3 = torch.nn.BatchNorm2d(num_features=32)
        self.bn64_1 = torch.nn.BatchNorm2d(num_features=64)
        self.bn64_2 = torch.nn.BatchNorm2d(num_features=64)
        self.bn128_1 = torch.nn.BatchNorm2d(num_features=128)
        self.bn128_2 = torch.nn.BatchNorm2d(num_features=128)
        self.bn128_3 = torch.nn.BatchNorm2d(num_features=128)
        self.bn128_4 = torch.nn.BatchNorm2d(num_features=128)
        self.bn256 = torch.nn.BatchNorm2d(num_features=256)
        self.bn512 = torch.nn.BatchNorm2d(num_features=512)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv64to128 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.conv32to128 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.conv128to32_1 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_2 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_3 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_4 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv64to32 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1)
        self.conv32to256 = torch.nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1, stride=1)
        self.conv32to512 = torch.nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1)
        self.conv256to128 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv512to128 = torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=1)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.softmax = torch.nn.Softmax(dim=1)

        self.handle_batchnorm2d32 = torch.nn.BatchNorm2d(num_features=32)
        self.handle_relu = torch.nn.ReLU()
        self.handle_conv2d32to128 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.handle_batchnorm2d128 = torch.nn.BatchNorm2d(num_features=128)
        self.handle_conv2d128to32 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.handle_dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):

        # 1st block
        x = self.conv1(x)
        x = self.bn64_1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 2nd block
        x = self.bn64_2(x)
        x = self.relu(x)
        x = self.conv64to128(x)
        x = self.bn128_1(x)
        x = self.relu(x)
        x = self.conv128to32_1(x)
        x = self.dropout(x)

        # nothing here

        x = self.bn32_1(x)
        x = self.relu(x)
        x = self.conv32to128(x)
        x = self.pool2_1(x)

        # 3rd block
        x = self.bn128_2(x)
        x = self.relu(x)
        x = self.conv128to32_2(x)
        x = self.dropout(x)


        # repeat 1 time block
        y1 = self.handle_batchnorm2d32(x)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d32to128(y1)
        y1 = self.handle_batchnorm2d128(y1)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d128to32(y1)
        y1 = self.handle_dropout(y1)

        x = torch.cat([x, y1], dim=-1)
        # repeat 1 time block

        x = self.bn32_2(x)
        x = self.relu(x)
        x = self.conv32to256(x)
        x = self.pool2_2(x)

        # 4th block
        x = self.bn256(x)
        x = self.relu(x)
        x = self.conv256to128(x)
        x = self.bn128_3(x)
        x = self.relu(x)
        x = self.conv128to32_3(x)
        x = self.dropout(x)

        # repeat 3 times block
        y1 = self.handle_batchnorm2d32(x)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d32to128(y1)
        y1 = self.handle_batchnorm2d128(y1)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d128to32(y1)
        y1 = self.handle_dropout(y1)

        y2 = torch.cat([x, y1], dim=-1)
        y2 = self.handle_batchnorm2d32(y2)
        y2 = self.handle_relu(y2)
        y2 = self.handle_conv2d32to128(y2)
        y2 = self.handle_batchnorm2d128(y2)
        y2 = self.handle_relu(y2)
        y2 = self.handle_conv2d128to32(y2)
        y2 = self.handle_dropout(y2)

        y3 = torch.cat([x, y1, y2], dim=-1)
        y3 = self.handle_batchnorm2d32(y3)
        y3 = self.handle_relu(y3)
        y3 = self.handle_conv2d32to128(y3)
        y3 = self.handle_batchnorm2d128(y3)
        y3 = self.handle_relu(y3)
        y3 = self.handle_conv2d128to32(y3)
        y3 = self.handle_dropout(y3)

        x = torch.cat([x, y1, y2, y3], dim=-1)
        # repeat 3 times block

        x = self.bn32_3(x)
        x = self.relu(x)
        x = self.conv32to512(x)
        x = self.pool2_3(x)

        # 5th block
        x = self.bn512(x)
        x = self.relu(x)
        x = self.conv512to128(x)
        x = self.bn128_4(x)
        x = self.relu(x)
        x = self.conv128to32_4(x)
        x = self.dropout(x)

        # repeat 2 times block
        y1 = self.handle_batchnorm2d32(x)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d32to128(y1)
        y1 = self.handle_batchnorm2d128(y1)
        y1 = self.handle_relu(y1)
        y1 = self.handle_conv2d128to32(y1)
        y1 = self.handle_dropout(y1)

        y2 = torch.cat([x, y1], dim=-1)

        y2 = self.handle_batchnorm2d32(y2)
        y2 = self.handle_relu(y2)
        y2 = self.handle_conv2d32to128(y2)
        y2 = self.handle_batchnorm2d128(y2)
        y2 = self.handle_relu(y2)
        y2 = self.handle_conv2d128to32(y2)
        y2 = self.handle_dropout(y2)

        x = torch.cat([x, y1, y2], dim=-1)
        # repeat 2 times block

        x = self.avgpool(x)
        x = self.softmax(x)
        return x


def go():
    # model121 = DenseNet([6, 12, 24, 16])
    # density: [1, 2, 4, 3], i.e. [0, 1, 3, 2].
    model = DenseNet()
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    return model


go()
