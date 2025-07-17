from pepedp.scripts.archs.blocklines import calculate_image_blockiness
from pepedp.scripts.utils.objects import IQANode


class BlockinessThread(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        thread: float = 0.5,
        median_thread=0,
        move_folder: str | None = None,
    ):
        super().__init__(
            img_dir, batch_size, thread, median_thread, move_folder, None, reverse=True
        )
        self.model = calculate_image_blockiness

    def forward(self, images):
        return self.model(images)
