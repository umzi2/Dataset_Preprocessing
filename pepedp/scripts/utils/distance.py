import torch
import torch.nn.functional as F


def cosine_dist(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor | float:
    """
    Поддерживает сравнение:
    - [D] и [D]
    - [1, D] и [1, D]
    - [N, D] и [M, D] -> [N, M]
    """
    emb1 = emb1.unsqueeze(0) if emb1.ndim == 1 else emb1  # [D] → [1, D]
    emb2 = emb2.unsqueeze(0) if emb2.ndim == 1 else emb2

    emb1_norm = F.normalize(emb1, dim=-1)
    emb2_norm = F.normalize(emb2, dim=-1)

    sim = emb1_norm @ emb2_norm.T  # [N, D] @ [D, M] = [N, M]
    dist = 1 - sim

    # if dist.numel() == 1:
    #     return dist.item()
    return dist


def euclid_dist(
    emb1: torch.Tensor, emb2: torch.Tensor, p: float = 2.0
) -> torch.Tensor | float:
    """
    Поддерживает сравнение:
    - [D] и [D]
    - [1, D] и [1, D]
    - [N, D] и [M, D] -> [N, M]
    """
    emb1 = emb1.unsqueeze(0) if emb1.ndim == 1 else emb1  # [D] → [1, D]
    emb2 = emb2.unsqueeze(0) if emb2.ndim == 1 else emb2

    d = torch.cdist(emb1, emb2, p=p)
    # if d.numel() == 1:
    #     return d.item()
    return d
