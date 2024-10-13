from enum import Enum

from src.scripts.aniiqa import AniIQAThread
from src.scripts.fast_hyper import HyperThread

class ThreadAlg(Enum):
    HIPERIQA = HyperThread
    ANIIQA = AniIQAThread
