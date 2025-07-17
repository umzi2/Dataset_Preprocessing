from enum import Enum
from pepedp.scripts.iqa import (
    HyperThread,
    AnIQAThread,
    TopIQThread,
    BlockinessThread,
    IC9600Thread,
)


class ThreadAlg(Enum):
    HIPERIQA = HyperThread
    ANIIQA = AnIQAThread
    TOPIQ = TopIQThread
    BLOCKINESS = BlockinessThread
    IC9600 = IC9600Thread
