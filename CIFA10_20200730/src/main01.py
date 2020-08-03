import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.ExampleNet import ExampleNet
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_map(y_test, y_pre):
    map_list = []
    for i in range(10):
        test = y_test == i
        pre = y_pre == i
        map_list.append(average_precision_score(test, pre))
    return sum(map_list) / len(map_list)


class Train:
    def __init__(self):
        # 可视化工具
        # 启用tensorboard:tensorboard --logdir=logs
        # http://localhost:6006/
        self.summaryWriter = SummaryWriter("./logs")

        # 获取训练数据
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.dataset = datasets.CIFAR10("D:\datasets", train=True, download=True, transform=transform_train)
        self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=True, drop_last=True)

        # 获取测试数据
        self.test_dataset = datasets.CIFAR10("D:\datasets", train=False, download=True, transform=transform_train)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=512, shuffle=True, drop_last=True)

        # 创建模型方式一
        # self.net = ExampleNet()
        # self.net.to(DEVICE)
        # self.net.load_state_dict(torch.load("checkpoint/2020-7-31-14-34-58.t"))

        # 创建模型方式二
        self.net = ExampleNet()
        if os.path.exists("models/net.pth"):
            self.net = torch.load("models/net.pth")
        self.net.to(DEVICE)

        # 损失函数
        self.loss_fn = nn.MSELoss()

        # 优化器
        self.optim = optim.Adam(self.net.parameters())

        # 训练代码

    def __call__(self):
        for epoch in range(20):
            loss_sum = 0.
            for i, (img, tage) in enumerate(self.dataloader):
                tage = torch.nn.functional.one_hot(tage, num_classes=10).float()
                # tage = torch.zeros(tage.size(0), 10).scatter_(1, tage.view(-1, 1), 1).to(DEVICE)
                img, tage = img.to(DEVICE), tage.to(DEVICE)

                y = self.net(img)
                loss = self.loss_fn(y, tage)
                print(loss)

                # 训练三件套
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.cpu().detach().item()
                print(loss_sum)
            avg_loss = loss_sum / len(self.dataloader)

            # 验证
            sum_score = 0.
            test_loss_sum = 0.
            y_test = []
            y_pred = []
            for i, (img, tage) in enumerate(self.test_dataloader):
                tage = torch.nn.functional.one_hot(tage, 10).float()
                # tage = torch.zeros(tage.size(0), 10).scatter_(1, tage.view(-1, 1), 1).to(DEVICE)
                img, tage = img.to(DEVICE), tage.to(DEVICE)
                test_y = self.net(img)
                loss = self.loss_fn(test_y, tage)
                test_loss_sum += loss.cpu().item()
                # 将one-hot转为标签值
                pre_tage = torch.argmax(test_y, dim=1)
                label_tage = torch.argmax(tage, dim=1)
                y_test.extend(label_tage.cpu().detach().numpy())
                y_pred.extend(pre_tage.cpu().detach().numpy())
                sum_score += torch.sum(torch.eq(pre_tage, label_tage).float())

            map_1 = count_map(np.array(y_test), np.array(y_pred))

            score = sum_score / len(self.test_dataset)
            test_avg_loss = test_loss_sum / len(self.test_dataloader)
            self.summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
            self.summaryWriter.add_scalar("score", score, epoch)
            self.summaryWriter.add_scalar("mAP", map_1, epoch)

            print(epoch, "train_loss:", avg_loss, "test_loss:", test_avg_loss, "score:", score.item())
            # 保存网络和参数，两种方式
            t = time.localtime(time.time())
            t1 = f"{t[0]}-{t[1]}-{t[2]}-{t[3]}-{t[4]}-{t[5]}"
            torch.save(self.net.state_dict(), f"./checkpoint/{t1}.t")
            torch.save(self.net, "models/net.pth")


if __name__ == '__main__':
    train = Train()
    train()
