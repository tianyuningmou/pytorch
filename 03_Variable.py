# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by tianyuningmou

FILE:  03_Variable.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/15 下午1:52

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/15 下午1:52
"""

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b

# 自动求导
y.backward()

# 对x求导，结果y'=w
print(x.grad)
# 对w求导，结果y'=x
print(w.grad)
# 对b求导，结果y'=1
print(b.grad)

x0 = torch.randn(3)
x0 = Variable(x0, requires_grad=True)

y0 = x0 * 2
print(y0)

y0.backward(torch.FloatTensor([1, 0.1, 0.01]))

print(x0.grad)
print('\n')
