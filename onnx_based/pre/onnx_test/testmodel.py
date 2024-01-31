import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1: 输入通道1, 输出通道6, 卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层2: 输入通道6, 输出通道16, 卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1: 输入特征数16*5*5, 输出特征数120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2: 输入特征数120, 输出特征数84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3: 输入特征数84, 输出特征数10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 通过第一个卷积层，然后应用ReLU激活函数
        x = F.relu(self.conv1(x))
        # 最大池化层，窗口大小2x2
        x = F.max_pool2d(x, 2)
        # 通过第二个卷积层，然后应用ReLU激活函数
        x = F.relu(self.conv2(x))
        # 最大池化层，窗口大小2x2
        x = F.max_pool2d(x, 2)
        # 展平所有除批量大小以外的维度
        x = x.view(-1, self.num_flat_features(x))
        # 通过第一个全连接层，然后应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层，然后应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三个全连接层
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 创建模型实例
model = LeNet()
model.eval()
inp = torch.randn(1, 1, 32, 32)
torch.onnx.export(model, inp, "./testModel.ir")
