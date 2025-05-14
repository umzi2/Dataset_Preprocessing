import torch
from chainner_ext import resize, ResizeFilter
from src.embedding.convnext import convnext_small, convnext_large
from src.embedding.enum import EmbeddedModel
import torch.nn.functional as F


def enum_to_model(enum: EmbeddedModel):
    match enum:
        case EmbeddedModel.ConvNextS:
            return convnext_small()
        case EmbeddedModel.ConvNextL:
            return convnext_large()
        case EmbeddedModel.VITS:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
        case EmbeddedModel.VITB:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
        case EmbeddedModel.VITL:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").eval()
        case EmbeddedModel.VITG:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").eval()


class ImgToEmbedding:
    def __init__(
        self,
        model: EmbeddedModel = EmbeddedModel.ConvNextS,
        amp: bool = True,
        scale: int = 4,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.scale = scale
        self.amp = amp
        self.model = enum_to_model(model).to(self.device)
        self.vit = model in [
            EmbeddedModel.VITS,
            EmbeddedModel.VITB,
            EmbeddedModel.VITL,
            EmbeddedModel.VITG,
        ]

    @staticmethod
    def check_img_size(x):
        b, c, h, w = x.shape
        mod_pad_h = (14 - h % 14) % 14
        mod_pad_w = (14 - w % 14) % 14
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def img_to_tensor(self, x):
        if self.vit:
            return self.check_img_size(
                torch.tensor(x.transpose((2, 0, 1)))[None, :, :, :].to(self.device)
            )
        return torch.tensor(x.transpose((2, 0, 1)))[None, :, :, :].to(self.device)

    @torch.inference_mode()
    def __call__(self, x):
        if self.scale > 1:
            h, w = x.shape[:2]
            x = resize(
                x, (w // self.scale, h // self.scale), ResizeFilter.CubicCatrom, False
            )
        with torch.amp.autocast(self.device.__str__(), torch.float16, self.amp):
            x = self.img_to_tensor(x)
            return self.model(x)
