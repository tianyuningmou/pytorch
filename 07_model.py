# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by easyto

FILE:  07_model.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/16 下午2:18

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/16 下午2:18
"""

'''
在PyTorch中使用torch.save来保存模型的结构和参数，有两种保存方式：
    ①保存整个模型的结构信息和参数信息，保存的对象是模型model，即torch.save(model, './model.pth')
    ②保存模型的参数，保存的对象是模型的状态model.state_dict()，即torch.save(model.state_dict(), './model_state.pth')

加载模型也有两种方式与保存模型对应：
    ①加载完整的模型结构和参数信息，使用load_model = torch.load('model.pth')，在网络较大的时候加载的时间比较长，同时存储空间也比较大
    ②加载模型参数信息，需要先导入模型的结构，然后通过model.load_state_dict(torch.load('model_state.pth'))来导入
'''
