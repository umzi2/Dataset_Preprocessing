import torch.amp

from pepedp.scripts.archs.ICNet import ic9600
from pepedp.scripts.utils.objects import IQANode


class IC9600Thread(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        thread: float = 0.5,
        median_thread=0,
        move_folder: str | None = None,
    ):
        super().__init__(img_dir, batch_size, thread, median_thread, move_folder, None)
        self.model = ic9600().to(self.device)

    @torch.autocast("cuda", torch.float16)
    @torch.no_grad()
    def forward(self, images):
        return self.model.get_only_score(images)
