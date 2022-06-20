import unittest

import pretty_midi

from .data import MelodyTranscriptionExample, Note, iter_hooktheory, load_hooktheory_raw
from .utils import compute_checksum


class TestData(unittest.TestCase):
    def test_note(self):
        Note(0.0, 0)
        Note(0.0, 60)
        Note(0.0, 127)
        Note(0.0, 60, 1.0)
        with self.assertRaises(TypeError):
            Note(0, 60)
        with self.assertRaises(TypeError):
            Note(0.0, 60.0)
        with self.assertRaises(TypeError):
            Note(0.0, 60, 1)
        with self.assertRaises(ValueError):
            Note(-1.0, 60)
        with self.assertRaises(ValueError):
            Note(1.0, 60, 0.0)
        with self.assertRaises(ValueError):
            Note(0.0, -1)
        with self.assertRaises(ValueError):
            Note(0.0, 128)

    def test_melody_transcription_example(self):
        notes = [
            [0, 60, 1],
            [1, 62, 2],
            [2, 64, 3],
            [3, 65, 4],
            [4, 67, 5],
        ]
        melody = [Note(float(on), p, float(off)) for on, p, off in notes]
        segment_start = melody[0].onset
        segment_end = melody[-1].offset
        MelodyTranscriptionExample(segment_start, segment_end, [])
        e = MelodyTranscriptionExample(segment_start, segment_end, melody)

        # Check MIDI round trip equivalence
        e_bytes = e.to_midi()
        e_rt = MelodyTranscriptionExample.from_midi(e_bytes)
        self.assertEqual(e_rt.to_midi(), e_bytes)

        # Check no offsets
        melody_no_offsets = [Note(float(on), p) for on, p, off in notes]
        MelodyTranscriptionExample(segment_start, segment_end, melody_no_offsets)
        MelodyTranscriptionExample(segment_start, melody[-1].onset, melody_no_offsets)

        # Check error handling
        with self.assertRaises(TypeError):
            MelodyTranscriptionExample(int(segment_start), segment_end, melody)
        with self.assertRaises(TypeError):
            MelodyTranscriptionExample(segment_start, int(segment_end), melody)
        with self.assertRaises(TypeError):
            MelodyTranscriptionExample(segment_start, segment_end, [None])
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(-1.0, segment_end, melody)
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(segment_end, segment_start, melody)
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(melody[1].onset, segment_end, melody)
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(segment_start, melody[-2].onset, melody)
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(segment_start, melody[-1].onset, melody)
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(
                segment_start, segment_end, [Note(0.0, 60), Note(0.0, 60)]
            )
        with self.assertRaises(ValueError):
            MelodyTranscriptionExample(
                segment_start, segment_end, [Note(0.0, 60, 1.0), Note(0.5, 62, 1.5)]
            )

    def test_hooktheory(self):
        pretty_midi.pretty_midi.MAX_TICK = 1e8

        # Test Hooktheory w/ refined (default) alignments
        hooktheory_raw = load_hooktheory_raw()
        self.assertEqual(len(hooktheory_raw), 16373)
        hooktheory = list(iter_hooktheory())
        self.assertEqual(len(hooktheory), 16373)
        for e in hooktheory:
            e_bytes = e.to_midi()
            e_rt = MelodyTranscriptionExample.from_midi(e_bytes)
            self.assertEqual(e_rt.to_midi(), e_bytes)
        hooktheory_test = list(iter_hooktheory(split="TEST"))
        self.assertEqual(len(hooktheory_test), 1480)

        # Test Hooktheory w/ user (crude) alignments
        hooktheory_raw = load_hooktheory_raw(alignment="USER")
        self.assertEqual(len(hooktheory_raw), 20112)
        hooktheory = list(iter_hooktheory(alignment="USER"))
        self.assertEqual(len(hooktheory), 20112)
        for e in hooktheory:
            e_bytes = e.to_midi()
            e_rt = MelodyTranscriptionExample.from_midi(e_bytes)
            self.assertEqual(e_rt.to_midi(), e_bytes)
