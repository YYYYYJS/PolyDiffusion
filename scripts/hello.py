import os

import numpy as np
import torch
from glob import glob

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

torch.manual_seed(2024)


class CustomDataset(Dataset):

    def __init__(self, root, data, transformations=None):

        self.transformations = transformations
        self.im_paths = sorted(glob(f"{root}/{data}/*/*"))

        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        for idx, im_path in enumerate(self.im_paths):
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1

    def get_class(self, path):
        return os.path.dirname(path).split("/")[-1]

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):

        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.get_class(im_path)]

        if self.transformations is not None: im = self.transformations(im)

        return im, gt


def get_dls(root, transformations, bs, split=[0.9, 0.1], ns=4):
    ds = CustomDataset(root=root, data="train", transformations=transformations)
    ts_ds = CustomDataset(root=root, data="test", transformations=transformations)

    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = total_len - tr_len

    tr_ds, vl_ds = random_split(dataset=ds, lengths=[tr_len, vl_len])

    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=ns), DataLoader(vl_ds,
                                                                                                      batch_size=bs,
                                                                                                      shuffle=False,
                                                                                                      num_workers=ns), DataLoader(
        ts_ds, batch_size=1, shuffle=False, num_workers=ns)

    return tr_dl, val_dl, ts_dl, ds.cls_names


root = "S:/data/flowerdataset"
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])
tr_dl, val_dl, ts_dl, classes = get_dls(root=root, transformations=tfs, bs=32)

print(len(tr_dl))
print(len(val_dl))
print(len(ts_dl))
print(classes)

if __name__ == "__main__":

    for idx, batch in tqdm(enumerate(tr_dl)):
        print(batch[0].shape, batch[1].shape)
