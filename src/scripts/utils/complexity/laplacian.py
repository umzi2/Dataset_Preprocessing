import cv2
import numpy as np

from src.scripts.utils.complexity.object import BaseComplexity


class LaplacianComplexity(BaseComplexity):
    def __init__(self, median_blur: int = 1):
        super().__init__()
        self.median_blur = median_blur if median_blur % 2 == 1 else median_blur + 1

    @staticmethod
    def image_to_gray(image: np.ndarray):
        image = image.squeeze()
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def median_laplacian(self, image):
        if self.median_blur <= 1:
            return image
        elif self.median_blur <= 5:
            image = cv2.medianBlur(image, self.median_blur)
        else:
            image = (
                cv2.medianBlur((image * 255).astype(np.uint8), self.median_blur).astype(
                    np.float32
                )
                / 255
            )
        return image

    @staticmethod
    def type():
        return "Laplacian"

    @staticmethod
    def get_tile_comp_score(image, complexity, y, x, tile_size):
        img_tile = image[
            y : y + tile_size,
            x : x + tile_size,
        ]
        score = np.mean(
            complexity[
                y : y + tile_size,
                x : x + tile_size,
            ]
        )
        complexity[
            y : y + tile_size,
            x : x + tile_size,
        ] = -1.0
        return img_tile, complexity, score

    def __call__(self, img):
        img = self.image_to_gray(img)
        img = self.median_laplacian(img)

        return np.abs(cv2.Laplacian(img, -1))
