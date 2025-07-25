import math
import random

import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip
import torch
import os
import pickle
from PIL import Image
from torchvision import transforms


def reverse_in_pairs(lst):
    head = lst[0:2]
    tail = lst[2:]
    for i in range(0, len(tail) - 1, 2):
        tail[i], tail[i + 1] = tail[i + 1], tail[i]
    tail.reverse()

    return head + tail


def get_dis(a, b):
    dis = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return dis


def rearrangement(lst):
    min_index = 0
    min = get_dis(lst[min_index:min_index + 2], [0, 0])
    for i in range(2, len(lst), 2):
        l = get_dis(lst[i:i + 2], [0, 0])
        if l < min:
            min_index = i
            min = l
    return lst[min_index:] + lst[:min_index]


def data_enhancement(lst, obj_num, min_values, max_values):
    num_points = len(lst[0])
    for j in range(10):
        if 0 < j < obj_num:
            r = random.random()
            if r < 0.0:
                num_masks = 36
                mask_indices = np.random.choice(num_points, num_masks, replace=False)
                for i in range(num_masks):
                    a = min(min_values[j], 0.005)
                    b = min(1 - max_values[j], 0.005)
                    arr = np.random.uniform(-a, b, size=(2,)).tolist()
                    lst[j][mask_indices[i]][0] += arr[0]
                    lst[j][mask_indices[i]][1] += arr[1]

    return lst


class ForceHorizontalFlip:
    def __call__(self, img):
        return hflip(img)


class CustomDataset(Dataset):
    def __init__(self, image_folder, cond_folder, transform=None):
        super(CustomDataset, self).__init__()
        self.image_folder = image_folder
        self.label_folder = cond_folder
        self.transform = transform
        self.image_transforms = transforms.Compose([

            ForceHorizontalFlip()  # 强制水平翻转
        ])

        # 获取图像文件列表和描述文件列表
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(cond_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])
        with open(label_name, 'rb') as file:
            label = pickle.load(file)
        obj_num = label['num_selected'][0]

        cond = {'obj_class': label['obj_class'],
                'obj_contour': label['obj_contour'],
                'obj_num': label['num_selected'],
                'is_valid_obj': label['is_valid_obj'],
                'obj_bbox': label['obj_bbox'],
                }

        # t = label['text'][0][0:5]

        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)
        image_flip = self.image_transforms(image)

        is_flip = np.random.rand()

        tensor_list = cond['obj_contour'].view(1, 10, -1).tolist()
        # tensor_list = cond['obj_contour'].view(10, -1, 2).tolist()
        # lst = cond['obj_contour'].view(10, -1).tolist()
        # min_values = [min(row) for row in lst]
        # max_values = [max(row) for row in lst]
        # tensor_list = torch.tensor(data_enhancement(tensor_list, obj_num, min_values, max_values)).view(1, 10,
        #                                                                                                -1).tolist()

        for k in range(1, obj_num):
            tensor_list[0][k] = rearrangement(tensor_list[0][k])

        if is_flip < 0.5:
            arr = []
            cond['obj_num'] = torch.tensor(obj_num, dtype=torch.int)
            for i in range(10):
                center_x = cond['obj_bbox'][0][i][0] + 0.5 * cond['obj_bbox'][0][i][2]
                center_y = cond['obj_bbox'][0][i][1] + 0.5 * cond['obj_bbox'][0][i][3]
                cls = torch.tensor([[center_x, center_y]])
                temp = torch.tensor(tensor_list[0][i]).view(96, 2)
                arr.append(torch.cat((cls, temp)).unsqueeze(0))
            arr = torch.cat(arr)

            cond['obj_contour'] = arr

            return image, cond
        else:
            obj_contour_flip = [[]]

            obj_bbox_flip = cond['obj_bbox'].tolist()

            for i in range(10):
                if 0 < i <= obj_num:
                    w = obj_bbox_flip[0][i][2]
                    x = 1.0 - obj_bbox_flip[0][i][0] - w
                    obj_bbox_flip[0][i][0] = x
                    center_x = x + 0.5 * w
                    center_y = obj_bbox_flip[0][i][1] + 0.5 * obj_bbox_flip[0][i][3]
                    modified_list = [1 - value if index % 2 == 0 else value for index, value in
                                     enumerate(tensor_list[0][i])]

                    obj_contour_flip[0].append([center_x, center_y] + rearrangement(reverse_in_pairs(modified_list)))

                else:

                    tensor_list[0][i] = [0, 0] + tensor_list[0][i]
                    obj_contour_flip[0].append(tensor_list[0][i])

            cond['obj_num'] = torch.tensor(obj_num, dtype=torch.int)
            cond['obj_bbox'] = torch.tensor(obj_bbox_flip)
            cond['obj_contour'] = torch.tensor(obj_contour_flip).view(10, 97, -1)

            return image_flip, cond
