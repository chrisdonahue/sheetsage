import numpy as np

from ..assets import retrieve_asset
from .handcrafted import OAFMelSpec as Handcrafted
from .jukebox import Jukebox as _Jukebox


class Jukebox(_Jukebox):
    def __init__(self):
        super().__init__(num_layers=53, fp16=False)
