import unittest

import numpy as np

from ..assets import retrieve_asset
from .handcrafted import OAFMelSpec


class TestHandcrafted(unittest.TestCase):
    def test_oaf_transcription(self):
        mp3_path = retrieve_asset("TEST_MP3")
        rate, rep = OAFMelSpec()(mp3_path)
        self.assertAlmostEqual(rate, 31.25, places=2)
        self.assertEqual(rep.shape, (144, 229))
        ref = np.load(retrieve_asset("TEST_MP3_OAFMELSPEC_REF")).T
        err = np.abs(rep - ref)
        self.assertLess(err.mean(), 0.015)
