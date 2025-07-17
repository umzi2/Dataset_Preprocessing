import cv2
import numpy as np
import torch

from pepedp.scripts.archs.ICNet import ic9600
from pepedp.scripts.utils.complexity.object import BaseComplexity


class IC9600Complexity(BaseComplexity):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.arch = ic9600().to(self.device)

    @staticmethod
    def image_to_tensor(image: np.ndarray):
        image = image.squeeze()
        if image.ndim == 2:
            return torch.tensor(
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).transpose((2, 0, 1))
            )[None, :, :, :]
        return torch.tensor(image.transpose((2, 0, 1)))[None, :, :, :]

    @staticmethod
    def type():
        return "IC9600"

    # @staticmethod
    @torch.inference_mode()
    def get_tile_comp_score(self, image, complexity, y, x, tile_size):
        img_tile = image[
            y * 8 : y * 8 + tile_size,
            x * 8 : x * 8 + tile_size,
        ]
        score = (
            self.arch.score(
                complexity[1][
                    :,
                    :,
                    y : y + tile_size // 8,
                    x : x + tile_size // 8,
                ]
            )
            .detach()
            .cpu()
            .item()
        )
        complexity[0][
            y : y + tile_size // 8,
            x : x + tile_size // 8,
        ] = -1.0
        return img_tile, complexity, score

    @torch.inference_mode()
    def __call__(self, img):
        img = self.image_to_tensor(img).to(self.device)
        x_cat, cly_map = self.arch(img)
        return cly_map.detach().cpu().squeeze().numpy(), x_cat
