from abc import ABC, abstractmethod
import numpy as np


class BaseComplexity(ABC):
    @staticmethod
    @abstractmethod
    def type() -> str: ...

    @abstractmethod
    def get_tile_comp_score(
        self, image, complexity, y: int, x: int, tile_size: int
    ): ...

    @abstractmethod
    def __call__(self, img: np.ndarray): ...
