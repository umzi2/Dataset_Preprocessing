import os
import shutil
import torch
from torch.utils.data import DataLoader

from module import ImageDataset
from tqdm import tqdm

class Thread:
    def __init__(self,name,thread):
        self.name = name
        self.thread = thread
    def __repr__(self):
        return f"Thread(Name = {self.name}, Thread = {self.thread}\n)"


class ThreadList:
    def __init__(self):
        self.mass = []
    def append(self,thread):
        self.mass.append(thread)
    def extend(self,thread_list):
        self.mass.extend(thread_list)
    def sort(self,reverse:bool=False):
        self.mass.sort(key=lambda item: item.thread,reverse=reverse)
    def __iter__(self):
        return iter(self.mass)
    def __getitem__(self, index):
        return self.mass[index]
    def __len__(self):
        return len(self.mass)

class IQANode:
    def __init__(self,img_dir,batch_size:int=8,thread:float=0.5,median_thread=0,move_folder:str|None=None,transform = None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataset = ImageDataset(img_dir, self.device,transform)
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.img_dir:str = img_dir
        self.thread = thread
        self.move_folder = move_folder
        if move_folder is not None :
            import os
            os.makedirs(move_folder,exist_ok=True)
        if median_thread:
            self.median_thread = median_thread
            self.thread_list = ThreadList()
        else:
            self.thread_list = None
    def __call__(self):
        for images, filenames in tqdm(self.data_loader):
            iqa = self.forward(images)
            for index in range(iqa.shape[0]):
                file_name = filenames[index]
                iqa_value = iqa[index]

                if self.thread_list is None:
                    if iqa[index]>self.thread and self.move_folder:
                        shutil.move( #shutil.clone(
                            os.path.join(self.img_dir,file_name),
                            os.path.join(self.move_folder,file_name)
                        )
                    elif iqa[index]<self.thread and not self.move_folder:
                        os.remove(os.path.join(self.img_dir,file_name))
                else:
                    if iqa[index]>self.thread:
                        self.thread_list.append(Thread(name=file_name, thread=float(iqa_value)))
        if self.thread_list:
            self.thread_list.sort()
            clip_index = int(len(self.thread_list) * self.median_thread)
            if self.move_folder:
                for thread in self.thread_list[-clip_index:]:
                    file_name = thread.name
                    shutil.move(
                        os.path.join(self.img_dir, file_name),
                        os.path.join(self.move_folder, file_name)
                    )
            else:
                for thread in self.thread_list[:-clip_index]:
                    file_name = thread.name
                    shutil.move(
                        os.path.join(self.img_dir, file_name),
                        os.path.join(self.move_folder, file_name)
                    )

