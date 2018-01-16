# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by tianyuningmou

FILE:  06_optim.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/15 下午4:05

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/15 下午4:05
"""

'''
优化算法分为两大类：
    ①一阶优化算法
    ②二阶优化算法
'''

import torch

'''
Example:
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()
'''
