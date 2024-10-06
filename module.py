import numpy as np
import torch
from torch import Tensor


def empty_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def image2tensor(
    value: list[np.ndarray] | np.ndarray,
    out_type: torch.dtype = torch.float32,
) -> list[Tensor] | Tensor:
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 2:
            tensor = torch.from_numpy(img[None, ...])
        else:
            tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if tensor.dtype != out_type:
            tensor = tensor.to(out_type)

        return tensor

    if isinstance(value, list):
        return [_to_tensor(i) for i in value]
    else:
        return _to_tensor(value)

