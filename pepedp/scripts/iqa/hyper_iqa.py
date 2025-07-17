from pyiqa import create_metric

from pepedp.scripts.utils.objects import IQANode


class HyperThread(IQANode):
    def __init__(
        self,
        img_dir,
        batch_size: int = 8,
        thread: float = 0.5,
        median_thread=0,
        move_folder: str | None = None,
    ):
        super().__init__(img_dir, batch_size, thread, median_thread, move_folder, None)
        self.model = create_metric("hyperiqa", device=self.device)

    def forward(self, images):
        return self.model(images)
