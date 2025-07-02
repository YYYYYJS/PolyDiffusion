from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn.functional as F
from contour_diffusion.dataset.mask_data import MaskDataset

from contour_diffusion.Encoder import TransformerModel
import torch.nn as nn

label_folder_path = 'S:/data/coco/128/mask_data/poly'
bbox_folder_path = 'S:/data/coco/128/mask_data/bbox'
class_folder_path = 'S:/data/coco/128/mask_data/class'
# 创建数据集实例
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MaskDataset(label_folder_path, bbox_folder_path, class_folder_path, transform)
batch_size = 2048
epoch = 5


def get_dis(a, b):
    bsz = a.shape[0]
    a = a.reshape(-1, 2)
    b = b.reshape(-1, 2)
    dis = F.pairwise_distance(a, b).sum() / bsz
    return dis


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
model = TransformerModel(2, 185, 128, 128, 8, 8).to('cuda')
model.load_state_dict(torch.load('transformer_encoder.pt'))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-05,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
for i in range(epoch):
    for j, data in enumerate(data_loader):
        poly, bbox, x, class_label, mask = data
        bbox = bbox.view(bbox.shape[0], 2, 2)
        x = x.to('cuda')
        class_label = class_label.to('cuda', dtype=torch.int64)
        mask = mask.to('cuda')
        reconstructed_coords, box_coordinates, class_pre = model(x, class_label, mask)
        poly = poly.to('cuda')
        bbox = bbox.to('cuda')
        y = poly[:, 1:, :]
        loss2 = criterion(class_pre, class_label)
        print(loss2)
        loss = get_dis(y, reconstructed_coords) + get_dis(bbox, box_coordinates) + loss2
        print(f'epoch=={i + 1}---step=={j + 1}---loss=={loss}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
torch.save(model.state_dict(), 'transformer_encoder.pt')
