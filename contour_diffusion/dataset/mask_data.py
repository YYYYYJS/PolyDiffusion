import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import pickle


def create_mask_from_indices(mask_indices, seq_len):
    """
    根据给定需要被掩码的索引列表和序列长度,创建掩码张量
    Args:
        mask_indices (list或torch.Tensor): 一个包含需要被掩码索引的列表或张量
        seq_len (int): 序列长度

    Returns:
        torch.Tensor: 形状为(1, seq_len)的掩码张量
    """
    # 创建初始全False掩码张量
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    # 根据索引列表设置True
    mask[0, mask_indices] = True
    mask[0] = False
    mask = mask.repeat(seq_len, 1)
    return mask


class MaskDataset(Dataset):
    def __init__(self,  poly_folder, bbox_folder, class_folder, transform=None):
        super(MaskDataset, self).__init__()

        self.label_folder = poly_folder
        self.bbox_folder = bbox_folder
        self.class_folder = class_folder
        self.transform = transform

        # 获取图像文件列表和描述文件列表
        # self.image_files = sorted(os.listdir(mask_folder))
        self.label_files = sorted(os.listdir(poly_folder))
        self.bbox_files = sorted(os.listdir(bbox_folder))
        self.class_files = sorted(os.listdir(class_folder))

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        # image_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])
        bbox_name = os.path.join(self.bbox_folder, self.bbox_files[idx])
        class_name = os.path.join(self.class_folder, self.class_files[idx])

        bbox = torch.load(bbox_name)
        center = []
        center.append(bbox[0] + 0.5 * bbox[2])
        center.append(bbox[1] + 0.5 * bbox[3])
        center = torch.tensor(center).view(1, 2)
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        poly = torch.load(label_name).view(-1, 2)
        poly = torch.cat((center, poly), dim=0)
        x = poly.clone()
        num_points = len(poly)

        r = random.random()
        x = x.numpy()
        num_masks = 0
        if r < 0.2:
            num_masks = 20
        elif r < 0.3:
            num_masks = 30
        elif r < 0.4:
            num_masks = 40
        elif r < 0.5:
            num_masks = 50
        elif r < 0.6:
            num_masks = 60

        mask_indices = np.random.choice(num_points, num_masks, replace=False)
        mask = create_mask_from_indices(mask_indices, poly.shape[0])

        x = torch.tensor(x)

        # mask_img = Image.open(image_name).convert('L')
        class_ = torch.load(class_name)
        # mask_img = self.transform(mask_img)

        return poly, bbox, x, class_, mask


class CustomDataset(Dataset):
    def __init__(self, image_folder, cond_folder, transform=None):
        super(CustomDataset, self).__init__()
        self.image_folder = image_folder
        self.label_folder = cond_folder
        self.transform = transform

        # 获取图像文件列表和描述文件列表
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(cond_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label_name = os.path.join(self.label_folder, self.label_files[idx])

        with open(label_name, 'rb') as file:
            label = pickle.load(file)

        cond = {
            'obj_contour': label['obj_contour'].squeeze(0),
            'obj_num': torch.tensor(label['num_selected'], dtype=torch.int),
            'obj_bbox': label['obj_bbox'].squeeze(0),
            'obj_class': label['obj_class'].squeeze(0),

        }

        return cond
