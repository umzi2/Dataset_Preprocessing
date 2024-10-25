from enum import Enum
from src.scripts.iqa import HyperThread, AnIQAThread, TopIQThread
class ThreadAlg(Enum):
    HIPERIQA = HyperThread
    ANIIQA = AnIQAThread
    TOPIQ = TopIQThread