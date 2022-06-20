import random
import unittest

import pretty_midi

from .assets import retrieve_asset
from .data import MelodyTranscriptionExample, Note
from .eval import EVAL_TOLERANCE, eval_dataset, f1


class TestEval(unittest.TestCase):
    def test_f1(self):
        # Create reference
        ref_notes = [
            (4, 60, 5),
            (5, 62, 6),
            (6, 64, 7),
            (7, 65, 8),
        ]
        melody = [Note(float(on), p, float(off)) for on, p, off in ref_notes]
        ref_example = MelodyTranscriptionExample(
            melody[0].onset, melody[-1].offset, melody
        )
        ref_midi = ref_example.to_midi()
        self.assertEqual(f1(ref_midi, ref_midi), (1.0, 1.0, 1.0, 00))

        def notes_to_midi(notes):
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(0)
            midi.instruments = [instrument]
            for on, p, off in notes:
                instrument.notes.append(
                    pretty_midi.Note(start=on, end=off, pitch=p, velocity=100)
                )
            return midi

        self.assertEqual(f1(ref_midi, notes_to_midi(ref_notes)), (1.0, 1.0, 1.0, 0))

        # Test timing invariance
        self.assertEqual(
            f1(
                ref_midi,
                notes_to_midi(
                    [
                        (4 + EVAL_TOLERANCE + 1e-3, 60, 5),
                        (5, 62, 6),
                        (6, 64, 7),
                        (7 - EVAL_TOLERANCE - 1e-3, 65, 8),
                    ]
                ),
            ),
            (0.5, 0.5, 0.5, 0),
        )
        for _ in range(100):
            est_notes = []
            for on, p, off in ref_notes:
                jitter = (random.random() * EVAL_TOLERANCE * 2) - EVAL_TOLERANCE
                est_notes.append((on + jitter, p, off))
            self.assertEqual(f1(ref_midi, notes_to_midi(est_notes)), (1.0, 1.0, 1.0, 0))

        # Test offset invariance
        for _ in range(100):
            est_notes = []
            for on, p, _ in ref_notes:
                est_notes.append((on, p, on + random.random() * 10))
            self.assertEqual(f1(ref_midi, notes_to_midi(est_notes)), (1.0, 1.0, 1.0, 0))
        for _ in range(100):
            est_notes = []
            for on, p, _ in ref_notes:
                est_notes.append((on, p, on + 1e-3))
            self.assertEqual(f1(ref_midi, notes_to_midi(est_notes)), (1.0, 1.0, 1.0, 0))

    def test_eval(self):
        f1, detail = eval_dataset(
            retrieve_asset("RWC_RYY_MIDI"),
            retrieve_asset("RWC_RYY_MIDI"),
            return_detail=True,
        )
        self.assertEqual(f1, 1.0)
        self.assertEqual(len(detail), 10)

        self.assertAlmostEqual(
            eval_dataset(
                retrieve_asset("RWC_RYYVOX_MIDI"),
                retrieve_asset("RWC_RYY_MIDI"),
                allow_abstain=True,
            ),
            0.907,
            places=3,
        )

        with self.assertWarnsRegex(Warning, "Abstained"):
            self.assertAlmostEqual(
                eval_dataset(
                    retrieve_asset("RWC_RYY_MIDI"),
                    retrieve_asset("RWC_RYYVOX_MIDI"),
                    allow_abstain=True,
                ),
                0.907,
                places=3,
            )

        with self.assertRaisesRegex(Exception, "Abstaining not allowed"):
            eval_dataset(
                retrieve_asset("RWC_RYY_MIDI"), retrieve_asset("RWC_RYYVOX_MIDI")
            )
