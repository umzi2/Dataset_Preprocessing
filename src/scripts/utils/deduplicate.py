import os

import torch

from src.embedding.embedding_class import ImgToEmbedding
from src.scripts.utils.distance import euclid_dist
from pepeline import read, ImgColor, ImgFormat


def create_embedd(
    img_folder: str,
    embedd_pth: str = "embedd.pth",
    embedder: ImgToEmbedding = ImgToEmbedding(),
):
    embedded = {}
    for img_name in os.listdir(img_folder):
        img = read(os.path.join(img_folder, img_name), ImgColor.RGB, ImgFormat.F32)
        embedded[img_name] = embedder(img).detach().cpu()
    torch.save(embedded, embedd_pth)


def duplicate_list(embedd_pth: str = "embedd.pth", dist_fn=euclid_dist, threshold=1):
    embeddings = torch.load(embedd_pth)
    names = list(embeddings.keys())
    used = set()
    clusters = []

    for i, name in enumerate(names):
        if name in used:
            continue

        cluster = [name]
        used.add(name)
        base_vec = embeddings[name]

        for j in range(i + 1, len(names)):
            other_name = names[j]
            if other_name in used:
                continue

            other_vec = embeddings[other_name]
            # dist = torch.norm(base_vec - other_vec, p=2).item()
            dist = dist_fn(base_vec, other_vec)

            if dist < threshold:
                cluster.append(other_name)
                used.add(other_name)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
