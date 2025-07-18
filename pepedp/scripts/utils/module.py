import os
from pepeline import read, ImgFormat
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, device, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.device = device

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = read(img_path, img_format=ImgFormat.F32)
        image = torch.tensor(image, dtype=torch.float32, device=self.device).permute(
            2, 0, 1
        )
        if self.transform:
            image = self.transform(image).to(self.device)

        return image, self.image_files[idx]
