import jittor
import jittor.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = jittor.nn.Conv2d(in_channels=1, kernel_size=5, out_channels=6)
        self.layer2 = jittor.nn.ReLU()
        self.layer3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = jittor.nn.Conv2d(in_channels=6, kernel_size=5, out_channels=16)
        self.layer5 = jittor.nn.ReLU()
        self.layer6 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7 = jittor.nn.Flatten()
        self.layer8 = jittor.nn.Linear(in_features=256, out_features=120)
        self.layer9 = jittor.nn.ReLU()
        self.layer10 = jittor.nn.Linear(in_features=120, out_features=84)
        self.layer11 = jittor.nn.ReLU()
        self.layer12 = jittor.nn.Linear(in_features=84, out_features=10)

    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x


def go():
    model = LeNet()
    x = jittor.randn(3, 1, 28, 28)
    y = model(x)
    return model
