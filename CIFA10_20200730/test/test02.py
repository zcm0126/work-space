import torch, thop
from torch import nn
import torch

if __name__ == '__main__':
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv = nn.Conv2d(3, 16, 3, 2, padding=0)

        def forward(self, x):
            return self.conv(x)


    conv = MyNet()
    x = torch.randn(1, 3, 16, 16)
    flops, params = thop.profile(conv, (x,))
    print(flops, params)
