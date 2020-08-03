import torch
import torch.nn.functional as F

a = F.one_hot(torch.arange(0, 5) % 3, num_classes=6)
print(torch.arange(0,5)%3)
print(a)