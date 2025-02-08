import os
import numpy as np
from pepeline import read, save, cvt_color, CvtType, best_tile, ImgColor, ImgFormat
import cv2
from tqdm.contrib.concurrent import process_map, thread_map
from tqdm import tqdm
from chainner_ext import resize, ResizeFilter

from src.enum import ProcessType


class BestTile:
    """
    Class for processing images to find and save the best tile with the highest Laplacian intensity.

    Attributes:
    - `in_folder` (str): Input folder containing images.
    - `out_folder` (str): Output folder to save processed images.
    - `tile_size` (int): Size of the tile to extract. Default is 1024.
    - `process_type` (str): Type of processing ('thread', 'process', or 'sequential'). Default is 'thread'.
    - `all_images` (list): List of all image filenames in the input folder.

    Methods:
    - `process(img_name: str)`: Process a single image to find and save the best tile.
    - `run()`: Run the processing on all images using the specified processing type.
    """

    def __init__(
        self,
        in_folder: str,
        out_folder: str,
        tile_size: int = 512,
        process_type: ProcessType = ProcessType.THREAD,
        scale: int = 1,
        dynamic_n_tiles: bool = True,
        median_blur: int = 0,
        laplacian_thread: float = 0,
        image_gray: bool = False,
    ):
        """
        Initialize the BestTile class.
        """
        self.scale = scale
        self.in_folder = in_folder
        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)
        self.out_list = os.listdir(out_folder)
        self.tile_size = tile_size
        self.all_images = os.listdir(in_folder)
        self.process_type = process_type
        self.dynamic_n_tiles = dynamic_n_tiles
        self.median_blur = median_blur if median_blur % 2 == 1 else median_blur + 1
        self.laplacian_thread = laplacian_thread
        self.image_gray = image_gray

    def save_result(self, img, img_name) -> None:
        save(img, os.path.join(self.out_folder, img_name))

    def get_tile(self, img, laplacian_abs):
        img_shape = laplacian_abs.shape
        if self.scale > 1:
            laplacian_abs_r = resize(
                laplacian_abs,
                (img_shape[1] // self.scale, img_shape[0] // self.scale),
                ResizeFilter.Linear,
                False,
            ).squeeze()
            left_up_cord = best_tile(laplacian_abs_r, self.tile_size // self.scale)
            left_up_cord = [index * self.scale for index in left_up_cord]
        else:
            left_up_cord = best_tile(laplacian_abs, self.tile_size)
        img = img[
            left_up_cord[0] : left_up_cord[0] + self.tile_size,
            left_up_cord[1] : left_up_cord[1] + self.tile_size,
        ]
        laplacian_abs[
            left_up_cord[0] : left_up_cord[0] + self.tile_size,
            left_up_cord[1] : left_up_cord[1] + self.tile_size,
        ] = -1.0
        return img, laplacian_abs

    def image_to_gray(self, image):
        if self.image_gray:
            return image
        return cvt_color(image, CvtType.RGB2GrayBt2020)

    def read_img(self, img_name):
        image = read(
            os.path.join(self.in_folder, img_name),
            ImgColor.GRAY if self.image_gray else ImgColor.RGB,
            ImgFormat.F32,
        )
        return image

    @staticmethod
    def laplacian_abs(image):
        return np.abs(cv2.Laplacian(image, -1))

    def median_laplacian(self, image):
        if self.median_blur <= 5:
            image = cv2.medianBlur(image, self.median_blur)
        else:
            image = (
                cv2.medianBlur((image * 255).astype(np.uint8), self.median_blur).astype(
                    np.float32
                )
                / 255
            )
        return self.laplacian_abs(image)

    def laplacian_image(self, image):
        img_gray = self.image_to_gray(image)
        if self.median_blur:
            return self.median_laplacian(img_gray)
        else:
            return self.laplacian_abs(img_gray)

    def process(self, img_name: str):
        """
        Process a single image to find and save the best tile.

        Args:
        - `img_name` (str): Name of the image file to process.
        """
        try:
            if img_name.split(".")[0] + ".png" in self.out_list:
                return
            img = self.read_img(img_name)
            img_shape = img.shape
            if img_shape[0] < self.tile_size or img_shape[1] < self.tile_size:
                return
            result_name = ".".join(img_name.split(".")[:-1]) + ".png"
            if img_shape[0] == self.tile_size and img_shape[1] == self.tile_size:
                self.save_result(img, result_name)
                return
            laplacian_abs = self.laplacian_image(img)
            if (
                img_shape[0] * img_shape[1] > self.tile_size**2 * 4
                and self.dynamic_n_tiles
            ):
                for i in range(
                    (img_shape[0] * img_shape[1]) // (self.tile_size**2 * 2)
                ):
                    tile, laplacian_abs = self.get_tile(img, laplacian_abs)
                    if self.laplacian_thread:
                        laplacian_tile = np.mean(self.laplacian_image(tile))
                        if laplacian_tile < self.laplacian_thread:
                            break
                    self.save_result(
                        tile, ".".join(img_name.split(".")[:-1]) + f"_{i}" + ".png"
                    )
            else:
                tile, laplacian_abs = self.get_tile(img, laplacian_abs)
                if self.laplacian_thread:
                    laplacian_tile = np.mean(self.laplacian_image(tile))
                    if laplacian_tile < self.laplacian_thread:
                        return
                self.save_result(tile, result_name)
        except Exception as e:
            print(img_name, "\n", e)

    def run(self):
        """
        Run the processing on all images using the specified processing type.
        """
        if self.process_type == ProcessType.THREAD:
            thread_map(self.process, self.all_images)
        elif self.process_type == ProcessType.PROCESS:
            process_map(self.process, self.all_images)
        else:
            for img_name in tqdm(self.all_images):
                self.process(img_name)
