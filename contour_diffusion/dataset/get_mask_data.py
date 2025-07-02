import copy
import os.path
import cv2
import numpy as np
from contour_diffusion.dataset.mask_data import CustomDataset
import torch
from torch.utils.data import DataLoader


def get_mask(data, size):
    batch_size = data['obj_num'].shape[0]

    res = []
    for i in range(batch_size):
        for j in range(1, int(data['obj_num'][i])):
            mask1 = np.zeros((size, size), dtype=np.uint8)
            seg1 = copy.deepcopy(data['obj_contour'][i][j])
            seg1[0::2] = seg1[0::2] * size
            seg1[1::2] = seg1[1::2] * size
            seg1 = seg1.to('cpu')
            seg1 = np.array(seg1).reshape(-1, 2)  # [n_points, 2]
            cv2.fillPoly(mask1, seg1.astype(np.int32)[np.newaxis, :, :], 1)

            res.append(mask1)

    return res


image_folder_path = 'S:/data/coco/128/full_image/images'
label_folder_path = 'S:/data/coco/128/full_image/conds'
# 创建数据集实例
dataset = CustomDataset(image_folder_path, label_folder_path)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

poly_save_path = 'S:/data/coco/128/mask_data/poly'
bbox_save_path = 'S:/data/coco/128/mask_data/bbox'
class_save_path = 'S:/data/coco/128/mask_data/class'
n = 1
for i, data in enumerate(data_loader):
    l = data['obj_num'][0]

    x = data["obj_contour"][0, :l + 1:]
    y = data["obj_bbox"][0, :l + 1:]
    z = data["obj_class"][0, :l + 1:]
    for j in range(l+1):
        path2 = os.path.join(poly_save_path, f'{n}.pt')
        path3 = os.path.join(bbox_save_path, f'{n}.pt')
        path4 = os.path.join(class_save_path, f'{n}.pt')

        torch.save(x[j], path2)
        torch.save(y[j], path3)
        torch.save(z[j], path4)
        n = n + 1
