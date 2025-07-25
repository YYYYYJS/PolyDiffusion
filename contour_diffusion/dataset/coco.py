#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
from collections import defaultdict
import random
import PIL
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import random
from contour_diffusion.dataset.util import image_normalize
from contour_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from transformers import CLIPTextModel, CLIPTokenizer
import math


def adjust(point_list, target_point):
    # 初始化最小距离和最近点的索引
    arr = []
    for j in range(len(point_list)):
        min_distance = float('inf')
        nearest_index = None

        # 循环遍历坐标列表
        for i in range(0, len(point_list[j]), 2):
            # 提取点的坐标
            x = point_list[j][i]
            y = point_list[j][i + 1]

            # 计算与目标点的距离
            distance = math.sqrt((x - target_point[0]) ** 2 + (y - target_point[1]) ** 2)

            # 如果距离比当前最小距离小，更新最小距离和最近点的索引
            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        if nearest_index is not None:
            arr1 = point_list[j][nearest_index:]
            arr2 = point_list[j][:nearest_index]
            new_point_list = arr1 + arr2
            arr.append(new_point_list)
    return arr


def generate_image_patch_boundary_points(resolution_to_attention):
    image_patch_boundary_points = {}
    for resolution in resolution_to_attention:
        image_size = 1
        # 计算图像块的大小
        block_size = image_size / resolution
        # 计算每个图像块内的点的间隔
        interval = block_size / 24
        # 生成每个图像块的边界点集合
        boundary_points = []
        for i in range(resolution):
            for j in range(resolution):
                x1 = j * block_size
                y1 = i * block_size

                # 生成边界点集合，按照左边框、下边框、右边框、上边框的顺序采样
                left_boundary = [(x1, y1 + y * interval) for y in range(25)]

                y1 = left_boundary[-1][1]
                bottom_boundary = [(x1 + (x + 1) * interval, y1) for x in range(24)]
                x1 = bottom_boundary[-1][0]
                right_boundary = [(x1, y1 - (y + 1) * interval) for y in range(24)]
                y1 = right_boundary[-1][1]
                top_boundary = [((x1 - (x + 1) * interval), y1) for x in range(23)]
                # 合并边界点集合
                block_boundary_points = left_boundary + bottom_boundary + right_boundary + top_boundary
                # 将相对坐标转换为绝对坐标并除以图像尺寸
                block_boundary_points = [(x / image_size, y / image_size) for x, y in block_boundary_points]
                # 将点集合展平为一维列表
                flat_block_boundary_points = [coord for point in block_boundary_points for coord in point]
                boundary_points.append(flat_block_boundary_points)
            # 存储在字典中
            image_patch_boundary_points['boundary_points_resolution{}'.format(resolution)] = boundary_points
    return image_patch_boundary_points


def interpolate_contour(contour, num_points):
    contour = np.squeeze(contour)
    contour_length = len(contour)
    if contour_length < num_points:
        # 计算需要插值的点数
        points_to_interpolate = num_points - contour_length
        # 生成要插值的索引
        interpolated_indexes = np.linspace(0, contour_length - 1, points_to_interpolate, dtype=np.int32)
        interpolated_contour = []
        for i in range(len(contour)):
            interpolated_contour.append(contour[i])
            if i in interpolated_indexes:
                if i == len(contour) - 1:
                    interpolated_contour.append(np.mean([contour[i], contour[0]], axis=0))
                else:
                    interpolated_contour.append(np.mean([contour[i], contour[i + 1]], axis=0))
            elif i in interpolated_indexes:
                interpolated_contour.append(contour[i])
        return interpolated_contour
    else:
        return contour


def get_contours(obj_contour):
    arr = []
    for i in range(len(obj_contour)):
        y = obj_contour[i][0]
        x = list(zip(y[::2], y[1::2]))
        if len(x) >= 48:
            x = interpolate_contour(x, 96)
        elif len(x) >= 24:
            x = interpolate_contour(x, 48)
            x = interpolate_contour(x, 96)
        elif len(x) >= 12:
            x = interpolate_contour(x, 24)
            x = interpolate_contour(x, 48)
            x = interpolate_contour(x, 96)
        elif len(x) >= 6:
            x = interpolate_contour(x, 12)
            x = interpolate_contour(x, 24)
            x = interpolate_contour(x, 48)
            x = interpolate_contour(x, 96)
        else:
            x = interpolate_contour(x, 6)
            x = interpolate_contour(x, 12)
            x = interpolate_contour(x, 24)
            x = interpolate_contour(x, 48)
            x = interpolate_contour(x, 96)
        x = [item for sublist in x for item in sublist]
        arr.append(x)
    return arr


def reverse_in_pairs(lst):
    head = lst[0:2]
    tail = lst[2:]
    for i in range(0, len(tail) - 1, 2):
        tail[i], tail[i + 1] = tail[i + 1], tail[i]
    tail.reverse()

    return head + tail


class CocoSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None, captions_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 max_num_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=1, max_objects_per_image=9, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, mode='train',
                 use_deprecated_stuff2017=False, deprecated_coco_stuff_ids_txt='', filter_mode='LostGAN',
                 use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=None
                 ):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - max_num_samples: If None use all images. Other wise only use images in the
          range [0, max_num_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to generator.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.use_adjust_point = True

        self.use_deprecated_stuff2017 = use_deprecated_stuff2017
        self.deprecated_coco_stuff_ids_txt = deprecated_coco_stuff_ids_txt
        self.mode = mode
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_num_samples = max_num_samples
        self.include_relationships = include_relationships
        self.filter_mode = filter_mode
        self.image_size = image_size
        self.min_object_size = min_object_size
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            # image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)
        if captions_json is not None and captions_json != '':
            with open(captions_json, 'r') as f:
                captions_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)
        self.image_id_to_caption = {}
        for image_data in captions_data['annotations']:
            image_id = image_data['image_id']
            image_caption = image_data['caption']
            if image_id not in self.image_id_to_caption:
                self.image_id_to_caption[image_id] = [image_caption]
            else:
                self.image_id_to_caption[image_id].append(image_caption)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        arr = []
        for object_data in instances_data['annotations']:
            if object_data['iscrowd'] == 0 and 6 <= len(object_data['segmentation'][0]) <= 192:
                arr.append(len(object_data['segmentation'][0]))

                image_id = object_data['image_id']
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other

                if self.filter_mode == 'LostGAN':
                    if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                        self.image_id_to_objects[image_id].append(object_data)
                elif self.filter_mode == 'SG2Im':
                    if box_ok and category_ok and other_ok:
                        self.image_id_to_objects[image_id].append(object_data)
                else:
                    raise NotImplementedError

        # Add object data from stuff
        stuff_data = None
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:

                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    # all object_data['iscrowd'] != 1, no need to filter iscrow
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # None for 184
        self.vocab['object_name_to_idx']['__null__'] = 184

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['__null__'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            flag = True
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs

            for i in range(num_objs):
                if self.image_id_to_objects[image_id][i]['category_id'] == 1:
                    flag = False
                    break

            if min_objects_per_image <= num_objs <= max_objects_per_image and flag:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        if self.use_deprecated_stuff2017:
            deprecated_image_ids = []
            with open(self.deprecated_coco_stuff_ids_txt, 'r') as f:
                for line in f.readlines():
                    image_id = line
                    deprecated_image_ids.append(
                        int(image_id)
                    )

            new_image_ids = []
            for image_id in self.image_ids:
                if int(image_id) in deprecated_image_ids:
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids

        # get 2048 test dataset
        if self.filter_mode == 'SG2Im':
            if self.mode == 'val':
                self.image_ids = self.image_ids[:1024]
            elif self.mode == 'test':
                self.image_ids = self.image_ids[-2048:]

        # get specific image ids or get specific number of images
        self.specific_image_ids = specific_image_ids
        if self.specific_image_ids:
            new_image_ids = []
            for image_id in self.specific_image_ids:
                if int(image_id) in self.image_ids:
                    new_image_ids.append(image_id)
                else:
                    print('image id: {} is not found in all image id list')
            self.image_ids = new_image_ids

        elif self.max_num_samples:
            self.image_ids = self.image_ids[:self.max_num_samples]

        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def get_init_meta_data(self, image_id):
        layout_length = self.max_objects_per_image + 2
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__']),
            'is_valid_obj': torch.zeros([layout_length]),
            'filename': self.image_id_to_filename[image_id].replace('/', '_').split('.')[0],
            'obj_contour': torch.zeros([layout_length, 192]),

        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        meta_data['obj_class'][0] = self.vocab['object_name_to_idx']['__image__']
        meta_data['is_valid_obj'][0] = 1.0

        return meta_data

    def load_image(self, image_id):
        with open(os.path.join(self.image_dir, self.image_id_to_filename[image_id]), 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.

        """
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        image = np.array(image, dtype=np.float32) / 255.0

        H, W, _ = image.shape
        num_obj = len(self.image_id_to_objects[image_id])
        obj_bbox = np.array([obj['bbox'] for obj in self.image_id_to_objects[image_id]])
        segmentations = [obj['segmentation'] for obj in self.image_id_to_objects[image_id]]
        obj_contour = np.array(segmentations, dtype=object)
        obj_contour = get_contours(obj_contour)

        if self.use_adjust_point:
            obj_contour = adjust(obj_contour, (0, 0))
        obj_class = np.array([obj['category_id'] for obj in self.image_id_to_objects[image_id]])
        is_valid_obj = [True for _ in range(num_obj)]

        # get meta data
        meta_data = self.get_init_meta_data(image_id=image_id)
        meta_data['width'], meta_data['height'] = W, H
        meta_data['num_obj'] = num_obj

        meta_data['text'] = self.image_id_to_caption[image_id]
        # filter invalid bbox
        # obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)

        # flip
        if self.left_right_flip:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)
            obj_contour = reverse_in_pairs(obj_contour)

        # random crop image and its bbox
        if self.use_MinIoURandomCrop:
            image, updated_obj_bbox, updated_obj_class, tmp_is_valid_obj = self.MinIoURandomCrop(image,
                                                                                                 obj_bbox[is_valid_obj],
                                                                                                 obj_class[
                                                                                                     is_valid_obj])

            tmp_idx = 0
            tmp_tmp_idx = 0
            for idx, is_valid in enumerate(is_valid_obj):
                if is_valid:
                    if tmp_is_valid_obj[tmp_idx]:
                        obj_bbox[idx] = updated_obj_bbox[tmp_tmp_idx]
                        tmp_tmp_idx += 1
                    else:
                        is_valid_obj[idx] = False
                    tmp_idx += 1

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape

        obj_bbox = torch.FloatTensor(obj_bbox[is_valid_obj])
        obj_contour = torch.FloatTensor(obj_contour)
        obj_class = torch.LongTensor(obj_class[is_valid_obj])
        obj_bbox[:, 0::2] = obj_bbox[:, 0::2] / W
        obj_bbox[:, 1::2] = obj_bbox[:, 1::2] / H

        obj_contour[:, 0::2] = obj_contour[:, 0::2] / W
        obj_contour[:, 1::2] = obj_contour[:, 1::2] / H

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox.shape[0]), num_selected)
        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        meta_data['obj_contour'][1:1 + num_selected] = obj_contour[selected_obj_idxs]
        meta_data['obj_contour'][0] = torch.tensor(
            generate_image_patch_boundary_points([1])['boundary_points_resolution1'])
        meta_data['obj_class'][1:1 + num_selected] = obj_class[selected_obj_idxs]
        meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] = num_selected

        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)

        return self.transform(image), meta_data


def coco_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_meta_data = defaultdict(list)
    all_imgs = []

    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        if key in ['obj_bbox', 'obj_class', 'is_valid_obj', 'obj_contour', 'obj_class_name_embedding'
                   ] or key.startswith(
            'labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_coco_dsets(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters
    params.use_deprecated_stuff2017 = False
    params.root_dir = 'E:/cocostuff'
    dataset = CocoSceneGraphDataset(
        mode=mode,
        filter_mode=params.filter_mode,
        stuff_only=params.stuff_only,
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_contour_object,
        min_object_size=params.min_object_size,
        min_objects_per_image=params.min_objects_per_image,
        max_objects_per_image=params.max_objects_per_image,
        instance_whitelist=params.instance_whitelist,
        stuff_whitelist=params.stuff_whitelist,
        include_other=params.include_other,
        include_relationships=params.include_relationships,
        use_deprecated_stuff2017=params.use_deprecated_stuff2017,
        deprecated_coco_stuff_ids_txt=os.path.join(params.root_dir, params[mode].deprecated_stuff_ids_txt),
        image_dir=os.path.join(params.root_dir, params[mode].image_dir),
        instances_json=os.path.join(params.root_dir, params[mode].instances_json),
        stuff_json=os.path.join(params.root_dir, params[mode].stuff_json),
        captions_json=os.path.join(params.root_dir, params[mode].captions_json),
        max_num_samples=params[mode].max_num_samples,
        left_right_flip=params[mode].left_right_flip,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids
    )

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print('%s dataset has %d images and %d objects' % (mode, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    return dataset
