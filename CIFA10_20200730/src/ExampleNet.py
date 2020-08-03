import torch
import torch.nn as nn


class ExampleNet(nn.Module):

    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv2d_3 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1)
        self.reLU_5 = nn.ReLU()
        self.maxPool2D_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_7 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1)
        self.reLU_9 = nn.ReLU()
        self.maxPool2D_11 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_6 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, )
        self.reLU_8 = nn.ReLU()
        self.linear_13 = nn.Linear(in_features=4 * 4 * 32, out_features=120)
        self.reLU_16 = nn.ReLU()
        self.linear_14 = nn.Linear(in_features=120, out_features=84)
        self.reLU_17 = nn.ReLU()
        self.linear_18 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x_para_1):
        x_conv2d_3 = self.conv2d_3(x_para_1)
        x_reLU_5 = self.reLU_5(x_conv2d_3)
        x_maxPool2D_4 = self.maxPool2D_4(x_reLU_5)
        x_conv2d_7 = self.conv2d_7(x_maxPool2D_4)
        x_reLU_9 = self.reLU_9(x_conv2d_7)
        x_maxPool2D_11 = self.maxPool2D_11(x_reLU_9)
        x_conv2d_6 = self.conv2d_6(x_maxPool2D_11)
        x_reLU_8 = self.reLU_8(x_conv2d_6)
        x_reshape_15 = torch.reshape(x_reLU_8, shape=(-1, 4 * 4 * 32))
        x_linear_13 = self.linear_13(x_reshape_15)
        x_reLU_16 = self.reLU_16(x_linear_13)
        x_linear_14 = self.linear_14(x_reLU_16)
        x_reLU_17 = self.reLU_17(x_linear_14)
        x_linear_18 = self.linear_18(x_reLU_17)
        return x_linear_18
