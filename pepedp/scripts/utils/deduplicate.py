import os
import shutil

import torch

from pepedp.embedding.embedding_class import ImgToEmbedding
from pepedp.scripts.utils.distance import euclid_dist
from pepeline import read, ImgColor, ImgFormat


def create_embedd(
    img_folder: str,
    embedder: ImgToEmbedding = ImgToEmbedding(),
):
    embedded = {}
    for img_name in os.listdir(img_folder):
        img = read(os.path.join(img_folder, img_name), ImgColor.RGB, ImgFormat.F32)
        embedded[img_name] = embedder(img).detach().cpu()
    return embedded


def filtered_pairs(
    embeddings, dist_func=euclid_dist, threshold: float = 1.5, device_str: str = None
):
    names = list(embeddings.keys())

    tensor_list = [embeddings[name] for name in names]
    E = torch.stack(tensor_list, dim=0)
    E = E.view(E.size(0), -1)

    device = (
        torch.device(device_str)
        if device_str
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    E = E.to(device)

    N = E.size(0)
    filtered_pairs = []

    for i in range(N):
        if i % 100 == 0 or i == N - 1:
            print(f"{i + 1} x {N}")

        anchor = E[i].unsqueeze(0)  # [1, D]
        compare_to = E[i + 1 :]  # [N - i - 1, D]
        if compare_to.size(0) == 0:
            continue

        dists = dist_func(anchor, compare_to).squeeze(0)  # [N - i - 1]
        mask = dists < threshold
        j_indices = torch.nonzero(mask).squeeze(1) + (i + 1)

        for idx, dist_val in zip(j_indices.tolist(), dists[mask].tolist()):
            if dist_val <= 0.01:
                print(dist_val, idx, names[idx], names[i])
            filtered_pairs.append((i, idx, dist_val))

    return {
        "names": names,
        "filtered_pairs": filtered_pairs,
    }


def move_duplicate_files(
    duplicates_dict: str, src_dir: str = "", dst_dir: str = ""
) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    names = duplicates_dict["names"]
    duplicates = duplicates_dict["filtered_pairs"]

    seen_indices = set()
    for i, j, _ in duplicates:
        seen_indices.add(i)
        seen_indices.add(j)

    moved_count = 0
    for idx in seen_indices:
        filename = names[idx]
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(src_path):
            print(f"[!] File not detect: {src_path}")
            continue

        if os.path.exists(dst_path):
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        shutil.move(src_path, dst_path)
        moved_count += 1

    print(f"✅ Moved {moved_count} duplicate files to: {dst_dir}")
