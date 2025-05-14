from enum import Enum

import torch

from src.embedding.convnext import convnext_small, convnext_large


class EmbeddedModel(Enum):
    ConvNextS = 0
    ConvNextL = 1
    VITS = 2
    VITB = 3
    VITL = 4
    VITG = 5
