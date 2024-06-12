import os

import numpy as np
from pepeline import read, save, cvt_color, CvtType, best_tile
import cv2
from tqdm.contrib.concurrent import process_map, thread_map


class BestTile:
    def __init__(self, in_folder: str, out_folder: str, tile_size: int = 1024, process_type: str = "thread"):
        self.in_folder = in_folder
        self.out_folder = out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        self.tile_size = tile_size
        self.all_images = os.listdir(in_folder)
        self.process_type = process_type

    def process(self, img_name:str):
        img = read(os.path.join(self.in_folder, img_name), 1, 0)
        img_shape = img.shape
        if img_shape[0]< self.tile_size or img_shape[1]<self.tile_size:
            return
        if img_shape[0] == self.tile_size or img_shape[1] == self.tile_size:
            save(img, os.path.join(self.out_folder, img_name))
            return
        img_gray = cvt_color(img, CvtType.RGB2GrayBt2020)
        laplacian_abs = np.abs(cv2.Laplacian(img_gray, -1))
        left_up_cord = best_tile(laplacian_abs, self.tile_size)
        save(img[left_up_cord[0]:left_up_cord[0] + self.tile_size, left_up_cord[1]:left_up_cord[1] + self.tile_size],
             os.path.join(self.out_folder, img_name))

    def run(self):
        if self.process_type == "thread":
            thread_map(self.process, self.all_images)
        elif self.process_type == "process":
            process_map(self.process, self.all_images)
        else:
            for img_name in self.all_images:
                self.process(img_name)
