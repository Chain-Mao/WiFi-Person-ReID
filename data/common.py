# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import torch
from .data_utils import read_image
import numpy as np


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = np.load(img_path)
        amplitude = img[:, 0::2, :].reshape(img.shape[0], -1)  # 从0开始，每隔一个取一个，即取出偶数通道
        phase = img[:, 1::2, :].reshape(img.shape[0], -1).transpose(1,0)  # 从1开始，每隔一个取一个，即取出奇数通道
        
        amplitude, amp_attention_mask = self.pad_attention_mask(amplitude)
        phase, pha_attention_mask = self.pad_attention_mask(phase)
        
        if self.transform is not None: 
            amplitude = self.transform(amplitude)
            phase = self.transform(phase)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "amplitude": amplitude,
            "phase": phase,
            "amp_attention_mask": torch.from_numpy(amp_attention_mask),
            "pha_attention_mask": torch.from_numpy(pha_attention_mask),
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

    def pad_attention_mask(self, input_tensor):
        # 获取输入张量的维度
        seq_length, feature_length = input_tensor.shape
        
        # 计算需要补零的维度
        pad_seq_length = 936 - seq_length    # 句子长度补零
        pad_feature_length = 936 - feature_length    # 特征长度补零
        
        # 在序列长度和特征长度后面补零
        # 注意，numpy.pad需要在每个轴上指定前后填充的数量，这里只需在后面填充，所以前面是0
        padded_tensor = np.pad(input_tensor, ((0, pad_seq_length), (0, pad_feature_length)), mode='constant', constant_values=0)
        
        # 创建attention_mask，初始全为0，长度为936
        attention_mask = np.zeros(936)
        
        # 将原始数据的对应位置设置为1
        attention_mask[:seq_length] = 1
        
        return padded_tensor, attention_mask
