import torch
import torch.nn.functional as F
def cosine_dist(emb1: torch.Tensor, emb2: torch.Tensor):
    emb1_norm = F.normalize(emb1, dim=-1)
    emb2_norm = F.normalize(emb2, dim=-1)
    return 1-F.cosine_similarity(emb1_norm, emb2_norm).item()
def euclid_dist(emb1:torch.Tensor,emb2:torch.Tensor):
    return torch.cdist(emb1,emb2).item()