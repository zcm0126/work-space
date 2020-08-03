import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.ExampleNet import ExampleNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 20
    batch_size = 512

    net = ExampleNet().to(device)
    criterion = nn.MSELoss(reduce=None, size_average=None)
    optimizer = optim.Adam(net.parameters(), weight_decay=0, amsgrad=False, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    summarywriter = SummaryWriter("./logs")

    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR10("datasets/", train=True, download=True, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testdataset = datasets.CIFAR10("datasets/", train=False, download=True, transform=transform_test)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    losses = []
    for i in range(epochs):
        net.train()
        print("epochs: {}".format(i))
        train_loss_sum = 0.
        test_loss_sum = 0.
        score_sum = 0.
        for j, (input, target) in enumerate(dataloader):
            input = input.to(device)
            output = net(input)
            target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1).to(device)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().detach().item()

            if j % 10 == 0:
                losses.append(loss.float())
                print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(dataloader), loss.float()))
                summarywriter.add_scalars("loss", {"train_loss": loss}, j)
                plt.clf()
                plt.plot(losses)
                plt.pause(0.01)

        with torch.no_grad():
            net.eval()
            correct = 0.
            total = 0.
            for input, target in testdataloader:
                input, target = input.to(device), target.to(device)
                output = net(input)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
                score_sum += accuracy
                target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1).to(device)
                test_loss = criterion(output, target)
                test_loss_sum += test_loss
            print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))

        torch.save(net, "models/net.pth")
