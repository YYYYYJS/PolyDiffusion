from torch.utils.data import Dataset
import os
import pickle


class CustomDataset(Dataset):
    def __init__(self, image_folder, cond_folder):
        super(CustomDataset, self).__init__()
        self.image_folder = image_folder
        self.label_folder = cond_folder

        # 获取图像文件列表和描述文件列表
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(cond_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])
        with open(label_name, 'rb') as file:
            cond = pickle.load(file)

        with open(img_name, 'rb') as file:
            img = pickle.load(file)

            return img, cond
