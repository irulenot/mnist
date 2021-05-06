import torch
from numpy import genfromtxt
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    train_numpy = genfromtxt('../digit-recognizer/train.csv', delimiter=',')
    test_numpy = genfromtxt('../digit-recognizer/test.csv', delimiter=',')
    train_numpy = train_numpy[2][1:].reshape((28, 28))
    test_numpy = test_numpy[2][1:].reshape((28, 28))
    train = torch.from_numpy(train_numpy[1:])
    test = torch.from_numpy(test_numpy[1:])

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(3, 6, 5)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #
    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = self.pool(F.relu(self.conv2(x)))
    #         x = x.view(-1, 16 * 5 * 5)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
    #
    # net = Net()