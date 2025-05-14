import os

import cv2
import numpy as np

from tqdm import tqdm

from src.embedding.embedding_class import ImgToEmbedding
from src.scripts.utils.distance import euclid_dist


class VideoToFrame:
    def __init__(
        self,
        embedder: ImgToEmbedding = ImgToEmbedding(),
        thread: float = 0.3,
        distance_fn=euclid_dist,
    ):
        self.embedder = embedder
        self.thread = thread
        self.distance_func = distance_fn

    def __call__(self, video_path, out_path):
        os.makedirs(out_path, exist_ok=True)
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ref = None
        n = 0
        with tqdm(total=total_frames) as pbar:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                if ref is None:
                    ref = self.embedder(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                        / 255.0
                    )
                else:
                    temp_embedd = self.embedder(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                        / 255.0
                    )
                    if self.distance_func(ref, temp_embedd) > self.thread:
                        cv2.imwrite(os.path.join(out_path, f"frame_{n}.png"), frame)
                        ref = temp_embedd
                n += 1
                pbar.update(1)

        capture.release()
