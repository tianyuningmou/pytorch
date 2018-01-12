# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by easyto

FILE:  02_Tensor.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/11 下午12:09

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/11 下午12:09
"""

import torch
import numpy

#Tensor方法默认创建的是FloatTensor数据类型
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print('a is {}'.format(a))
print('a size is {}'.format(a.size()))

b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print('b is : {}'.format(b))

# 全为0的Tensor
c = torch.zeros((3, 2))
print('zero tensor: {}'.format(c))

#取一个正态分布作为随机初始值
d = torch.randn((3, 2))
print('normal random is : {}'.format(d))

#以索引的方式改变Tensor的值
a[0, 1] = 100
print('changed a is : {}'.format(a))

#numpy转Tensor
numpy_b = b.numpy()
print('convssssssser to numpy is \n {}'.format(numpy_b))

#Tensor转numpy
e = numpy.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))

f_torch_e = torch_e.float()
print('change data type to float tensor: {}'.format(f_torch_e))
