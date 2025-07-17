import torch
from torchvision.transforms import transforms
from pepedp.scripts.utils.objects import IQANode


class AnIQAThread(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        thread: float = 0.5,
        median_thread=0,
        move_folder: str | None = None,
    ):
        compose = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        super().__init__(
            img_dir, batch_size, thread, median_thread, move_folder, compose
        )
        self.model = torch.hub.load(
            repo_or_dir="miccunifi/ARNIQA",
            source="github",
            model="ARNIQA",
            regressor_dataset="kadid10k",
        )
        self.model.eval().to(self.device)

    def forward(self, images):
        _, _, h, w = images.size()
        images_ds = transforms.Resize((h // 2, w // 2)).to(images)(images)
        with torch.inference_mode():
            iqa = self.model(
                images, images_ds, return_embedding=False, scale_score=True
            )
        return iqa
