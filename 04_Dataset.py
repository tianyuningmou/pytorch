# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved by tianyuningmou

FILE:  04_Dataset.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2018/1/15 下午2:27

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2018/1/15 下午2:27
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# 计算机视觉的数据读取类
from torchvision.datasets.folder import ImageFolder, default_loader
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data


dataiter = DataLoader(MyDataset, batch_size=32, shuffle=True, collate_fn=default_collate)

dset = ImageFolder(root='root_path', transform=None, loader=default_loader)
