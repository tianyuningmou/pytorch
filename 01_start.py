# -*- coding: utf-8 -*-

"""
Copyright () 2017

All rights reserved by tianyuningmou

FILE:  01_start.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2017/12/27 下午3:40

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2017/12/27 下午3:40
"""

from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
y = torch.rand(5, 3)
z = torch.rand(5, 3)
print(x, '\n', y, '\n', x.size(), '\n', y+z)

