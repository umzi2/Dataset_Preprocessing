import os
from pepeline import read, save, best_tile, ImgColor, ImgFormat
from tqdm.contrib.concurrent import process_map, thread_map
from tqdm import tqdm
from chainner_ext import resize, ResizeFilter

from src.enum import ProcessType
from src.scripts.utils.complexity.laplacian import LaplacianComplexity
from src.scripts.utils.complexity.object import BaseComplexity


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
        laplacian_thread: float = 0,
        image_gray: bool = False,
        func: BaseComplexity = LaplacianComplexity(),
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

        self.laplacian_thread = laplacian_thread
        self.image_gray = image_gray
        self.func = func
        if func.type() == "IC9600":
            self.process_type = ProcessType.FOR

    def save_result(self, img, img_name) -> None:
        save(img, os.path.join(self.out_folder, img_name))

    def get_tile(self, img, complexity):
        if self.func.type() != "Laplacian":
            left_up_cord = best_tile(complexity[0], self.tile_size // 8)
        elif self.scale > 1:
            img_shape = complexity.shape
            complexity_r = resize(
                complexity,
                (img_shape[1] // self.scale, img_shape[0] // self.scale),
                ResizeFilter.Linear,
                False,
            ).squeeze()
            left_up_cord = best_tile(complexity_r, self.tile_size // self.scale)
            left_up_cord = [index * self.scale for index in left_up_cord]
        else:
            left_up_cord = best_tile(complexity, self.tile_size)
        img, laplacian, score = self.func.get_tile_comp_score(
            img, complexity, left_up_cord[0], left_up_cord[1], self.tile_size
        )
        return img, laplacian, score

    def read_img(self, img_name):
        image = read(
            os.path.join(self.in_folder, img_name),
            ImgColor.GRAY if self.image_gray else ImgColor.RGB,
            ImgFormat.F32,
        )
        return image

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
            complexity = self.func(img)
            if (
                img_shape[0] * img_shape[1] > self.tile_size**2 * 4
                and self.dynamic_n_tiles
            ):
                for i in range(
                    (img_shape[0] * img_shape[1]) // (self.tile_size**2 * 2)
                ):
                    tile, laplacian_abs, score = self.get_tile(img, complexity)
                    if self.laplacian_thread:
                        if score < self.laplacian_thread:
                            break
                    self.save_result(
                        tile, ".".join(img_name.split(".")[:-1]) + f"_{i}" + ".png"
                    )
            else:
                tile, laplacian_abs, score = self.get_tile(img, complexity)
                if self.laplacian_thread:
                    if score < self.laplacian_thread:
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
