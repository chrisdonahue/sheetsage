import tempfile
import unittest

import librosa
import numpy as np
from scipy.io.wavfile import read as wavread

from ..assets import retrieve_asset
from ..utils import compute_checksum
from .jukebox import _CHUNK_FRAMES, _SAMPLE_RATE, Jukebox

_TEST_NUM_LAYERS = 36


class TestJukebox(unittest.TestCase):
    def test_singleton(self):
        Jukebox(num_layers=_TEST_NUM_LAYERS)
        with self.assertRaisesRegex(Exception, "initialized once"):
            Jukebox(num_layers=1)

    def test_decode_audio_strict(self):
        # NOTE: *Very* strict test to check the decoding stack against training config.
        # Versions: ffmpeg 4.3.2-0york0~18.04, librosa 0.7.2, resampy 0.2.2
        audio = Jukebox.decode_audio(retrieve_asset("TEST_MP3"))
        _, audio_ref = wavread(retrieve_asset("TEST_MP3_JUKEBOX_DECODE_REF"))
        self.assertTrue(np.array_equal(audio, audio_ref))

    def test_codify_audio(self):
        # NOTE: *Pretty* strict test to check VQVAE against training.
        # Versions: torch 1.4.0+cu101
        jukebox = Jukebox(num_layers=_TEST_NUM_LAYERS)
        _, audio = wavread(retrieve_asset("TEST_MP3_JUKEBOX_DECODE_REF"))
        audio_codified = jukebox.codify_audio(audio)
        self.assertEqual(audio_codified.shape, (8192,))
        self.assertEqual(np.unique(audio_codified).shape[0], 432)
        self.assertEqual(
            compute_checksum(str(audio_codified.tolist()).encode("utf-8")),
            "2569ffae9819a43a0e4ba29f3caa5ecab6d4b2a8e1b3fde0a65bbf6532a7e479",
        )
        self.assertEqual(np.unique(audio_codified[1580:]).tolist(), [653, 1489])

        audio_codified_nopad = jukebox._codify_audio(audio, pad=False)
        self.assertEqual(audio_codified_nopad.shape, (1580,))
        eq = audio_codified[: audio_codified_nopad.shape[0]] == audio_codified_nopad
        acc = eq.astype(np.float32).mean()
        self.assertAlmostEqual(acc, 0.9994, places=4)

    def test_lm_activations(self):
        jukebox = Jukebox(num_layers=_TEST_NUM_LAYERS)
        _, audio = wavread(retrieve_asset("TEST_MP3_JUKEBOX_DECODE_REF"))
        audio_codified = jukebox.codify_audio(audio)

        audio_activations = jukebox.lm_activations(audio_codified)
        self.assertEqual(audio_activations.shape, (8192, 4800))
        self.assertAlmostEqual(np.abs(audio_activations).mean(), 6.3359, places=4)
        self.assertAlmostEqual(
            np.abs(audio_activations).astype(np.float64).sum(), 249135949.1, places=1
        )

    def test_extract(self):
        jukebox = Jukebox(num_layers=_TEST_NUM_LAYERS, fp16=True)
        mp3_path = retrieve_asset("TEST_MP3")
        rate, audio_activations = jukebox(mp3_path)
        self.assertAlmostEqual(rate, 344.5, places=1)
        self.assertEqual(audio_activations.shape, (1580, 4800))
        self.assertAlmostEqual(np.abs(audio_activations).mean(), 1.7705, places=4)
        self.assertAlmostEqual(
            np.abs(audio_activations).astype(np.float64).sum(), 13425792.4, places=1
        )

    def test_edge_cases(self):
        jukebox = Jukebox(num_layers=_TEST_NUM_LAYERS)
        jukebox(
            retrieve_asset("YOUTUBE_ZqJiXLJs_Pg"),
            offset=281.66999999999996,
            duration=15.360000000000014,
        )

    def test_legacy(self):
        jukebox = Jukebox(num_layers=_TEST_NUM_LAYERS)
        audio = jukebox.decode_audio(retrieve_asset("TEST_JUKEBOX_LEGACY"))
        audio = audio[: 25 * _SAMPLE_RATE]
        codified_audio = jukebox._codify_audio(audio, window_size=25 * _SAMPLE_RATE)
        codified_audio = codified_audio[:_CHUNK_FRAMES]
        activations = jukebox.lm_activations(
            codified_audio, metadata_total_length_seconds=62
        )
        expected = np.load(retrieve_asset("TEST_JUKEBOX_LEGACY_REF"))
        recomputed = np.mean(activations, axis=0)
        err = np.abs(recomputed - expected).sum()
        self.assertLess(err, 0.01)
