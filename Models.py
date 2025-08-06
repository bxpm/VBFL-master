import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class Mnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 1, 28, 28)
		tensor = F.relu(self.conv1(tensor))
		tensor = self.pool1(tensor)
		tensor = F.relu(self.conv2(tensor))
		tensor = self.pool2(tensor)
		tensor = tensor.view(-1, 7*7*64)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Mnist_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.fc1 = nn.Linear(7 * 7 * 512, 1024)  # 28x28图像经过4次池化后变为28/2^2=7
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 10)  # 假设有10个类别
#
#     def forward(self, inputs):
#         tensor = inputs.view(-1, 1, 28, 28)
#
#         tensor = F.relu(self.bn1(self.conv1(tensor)))  # 28x28 -> 28x28
#         tensor = self.pool(tensor)  # 28x28 -> 14x14
#
#         tensor = F.relu(self.bn2(self.conv2(tensor)))  # 14x14 -> 14x14
#         tensor = self.pool(tensor)  # 14x14 -> 7x7
#
#         tensor = F.relu(self.bn3(self.conv3(tensor)))  # 7x7 -> 7x7
#         tensor = self.pool(tensor)  # 7x7 -> 3x3 (向下取整)
#
#         tensor = F.relu(self.bn4(self.conv4(tensor)))  # 3x3 -> 3x3
#         tensor = self.pool(tensor)  # 3x3 -> 1x1 (向下取整)
#
#         tensor = tensor.view(-1, 7 * 7 * 512)  # 如果池化后尺寸不符合，调整为正确的维度
#         tensor = F.relu(self.fc1(tensor))
#         tensor = self.dropout(tensor)
#         tensor = self.fc2(tensor)
#         return tensor
