import os
import shutil
import torch
from torch.utils.data import  DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from module import ImageDataset


class AniIQAThread:
    def __init__(self,img_dir,batch_size:int=8,thread:float=0.5,move_folder:str|None=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageDataset(img_dir, device, preprocess)

        self.img_dir = img_dir
        self.model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k")
        self.model.eval().to(device)

        self.data_loader = DataLoader(dataset,batch_size=batch_size)
        self.thread = thread
        self.move_folder = move_folder
        if not move_folder is None :
            os.makedirs(move_folder, exist_ok=True)

    def run(self):
        for images, filenames in tqdm(self.data_loader):
            _, _, h, w = images.size()
            images_ds = transforms.Resize((h//2,w//2)).to(images)(images)
            with torch.no_grad():
                iqa = self.model(images,images_ds, return_embedding=False, scale_score=True) < self.thread
            for index in range(iqa.shape[0]):
                file_name = filenames[index]
                if not iqa[index] and self.move_folder:
                    shutil.move(
                        os.path.join(self.img_dir,file_name),
                        os.path.join(self.move_folder,file_name)
                    )
                elif iqa[index] and not self.move_folder:
                    os.remove(os.path.join(self.img_dir,file_name))
