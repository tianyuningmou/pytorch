# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by tianyuningmou

FILE:  05_Module.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/15 下午3:29

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/15 下午3:29
"""

from torch import nn


class NetName(nn.Module):
    def __init__(self, other_arguments):
        super(NetName, self).__init__()
        self.conv1 = nn.Conv2d(in_channels='', out_channels='', kernel_size='')

    def forward(self, x):
        x = self.conv1(x)
        return x
