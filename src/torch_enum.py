from enum import Enum
from src.scripts.iqa import HyperThread, AnIQAThread, TopIQThread, BlockinessThread


class ThreadAlg(Enum):
    HIPERIQA = HyperThread
    ANIIQA = AnIQAThread
    TOPIQ = TopIQThread
    BLOCKINESS = BlockinessThread
