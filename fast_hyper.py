import os
import shutil
import torch
from torch.utils.data import  DataLoader
from pyiqa import create_metric
from tqdm import tqdm
from module import ImageDataset


class HyperThread:
    def __init__(self,img_dir,batch_size:int=8,thread:float=0.5,move_folder:str|None=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        dataset = ImageDataset(img_dir, device)

        self.img_dir = img_dir
        self.hyperiqa =  create_metric("hyperiqa",device=device)
        self.data_loader = DataLoader(dataset,batch_size=batch_size)
        self.thread = thread
        self.move_folder = move_folder
        if not move_folder is None and not os.path.exists(move_folder):
            os.makedirs(move_folder)

    def run(self):
        for images, filenames in tqdm(self.data_loader):
            iqa = self.hyperiqa(images) < self.thread
            for index in range(iqa.shape[0]):
                file_name = filenames[index]
                if not iqa[index] and self.move_folder:
                    shutil.move(
                        os.path.join(self.img_dir,file_name),
                        os.path.join(self.move_folder,file_name)
                    )
                elif iqa[index] and not self.move_folder:
                    os.remove(os.path.join(self.img_dir,file_name))
