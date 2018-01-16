# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by easyto

FILE:  09_PR.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/16 下午4:23

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/16 下午4:23
"""

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


# 使用torch.cat()函数来实现Tensor的拼接
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(W_target) + b_target[0]


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


# 定义模型
class PolyRegression(nn.Module):
    def __init__(self):
        super(PolyRegression, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


model = PolyRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epoch = 0

while True:
    # Get data
    batch_x, batch_y = get_batch()
    # Forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]
    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimizer.step()
    epoch += 1

    print('Epoch[{}], loss:{}'.format(epoch, loss.data[0]))

    if print_loss < 1e-3:
        break

plt.plot(batch_x.data.numpy(), batch_y.data.numpy())
plt.show()
