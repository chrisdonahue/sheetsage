import unittest

from scipy.io.wavfile import read as wavread

from .assets import retrieve_asset
from .beat_track import madmom


class TestBeatTrack(unittest.TestCase):
    def test_madmom(self):
        sr, wav = wavread(retrieve_asset("TEST_WAV"))
        expected_beats = [
            0.07,
            0.45,
            0.85,
            1.25,
            1.63,
            1.98,
            2.33,
            2.67,
            3.04,
            3.39,
            3.72,
            4.08,
            4.43,
        ]

        # Test madmom without auxiliary inputs
        first_downbeat, beats_per_bar, beats = madmom(sr, wav)
        self.assertEqual(first_downbeat, 3)
        self.assertEqual(beats_per_bar, 4)
        self.assertEqual(beats, expected_beats)

        # Restrict possible time signatures
        first_downbeat, beats_per_bar, beats = madmom(
            sr,
            wav,
            beats_per_bar=[1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(first_downbeat, 3)
        self.assertEqual(beats_per_bar, 4)
        self.assertEqual(beats, expected_beats)

        # Pass in fixed (and incorrect) time signature
        first_downbeat, beats_per_bar, beats = madmom(sr, wav, beats_per_bar=3)
        self.assertEqual(first_downbeat, 2)
        self.assertEqual(beats_per_bar, 3)
        self.assertEqual(beats, expected_beats)

        # Pass in first two seconds
        first_downbeat, beats_per_bar, beats = madmom(sr, wav[: int(2 * sr)])
        self.assertEqual(first_downbeat, 0)
        self.assertEqual(beats_per_bar, 3)
        self.assertEqual(beats, [0.07, 0.45, 0.85, 1.25, 1.63])

        # Pass in first second
        first_downbeat, beats_per_bar, beats = madmom(sr, wav[: int(1 * sr)])
        self.assertEqual(first_downbeat, 0)
        self.assertEqual(beats_per_bar, None)
        self.assertEqual(beats, [0.07, 0.45, 0.85])

        # Pass in nothing
        first_downbeat, beats_per_bar, beats = madmom(sr, wav[:0])
        self.assertEqual(first_downbeat, None)
        self.assertEqual(beats_per_bar, None)
        self.assertEqual(beats, [])
