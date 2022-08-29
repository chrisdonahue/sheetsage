import numpy as np

from ..assets import retrieve_asset
from .handcrafted import OAFMelSpecNorm
from .jukebox import Jukebox as _Jukebox


class Handcrafted(OAFMelSpecNorm):
    def __init__(self):
        moments = np.load(retrieve_asset("SHEETSAGE_V02_HANDCRAFTED_MOMENTS"))
        super().__init__(moments)


class Jukebox(_Jukebox):
    def __init__(self):
        super().__init__(num_layers=53, fp16=False)
