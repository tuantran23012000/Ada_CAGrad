import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class MultiLeNet(torch.nn.Module):
    def __init__(self, n_tasks):
        super(MultiLeNet, self).__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)
        self.fc2 = nn.Linear(50, 10)
    def shared_modules(self):
        return [self.conv1, self.conv2,
                self.fc1]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))
        task1 = self.fc2(x)
        task2 = self.fc2(x)
        return [task1, task2]

