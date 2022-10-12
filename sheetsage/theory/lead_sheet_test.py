import json
import unittest
from io import BytesIO

import numpy as np
import pretty_midi

from ..assets import retrieve_asset
from .internal import Harmony, KeyChanges, Melody, MeterChanges, TempoChanges
from .lead_sheet import LeadSheet


class TestLeadSheet(unittest.TestCase):
    def test_lead_sheet(self):
        ls = LeadSheet(
            MeterChanges((0, (4, 2, 2))),
            TempoChanges((0, (120,))),
            KeyChanges((0, (0, (2, 2, 1, 2, 2, 2)))),
            Harmony(
                (8, (5, (4, 3))),
                (12, (7, (4, 3))),
                (16, (0, (4, 3))),
            ),
            Melody(
                (8, 1, (0, 0)),
                (10, 1, (2, 0)),
                (12, 2, (4, 0)),
                (14, 2, (5, 0)),
                (16, 8, (7, 0)),
            ),
            32,
        )
        self.assertTrue(isinstance(ls, LeadSheet))
        self.assertTrue(isinstance(ls, tuple))
        self.assertEqual(LeadSheet(*ls), ls)

        # Make sure minimum length is one measure
        self.assertEqual(
            LeadSheet(
                MeterChanges((0, (4, 2, 2))),
                TempoChanges((0, (120,))),
                KeyChanges((0, (0, (2, 2, 1, 2, 2, 2)))),
                Harmony(),
                Melody(),
            )[-1],
            16,
        )

        # Test LilyPond output
        lily = ls.as_lily()
        lily_expected = r"""
#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        s16*8 f16*4 g16*4 c16*16
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 120
        r2 c''16 r16 d''16 r16 e''8 f''8 | g''2 r2
    }
}

>>

\version "2.18.2"
        """.strip()
        self.assertEqual(lily, lily_expected)

        # Test MIDI output
        midi = pretty_midi.PrettyMIDI(BytesIO(ls.as_midi()))
        self.assertEqual(len(midi.instruments), 3)
        [click, harmony, melody] = midi.instruments
        self.assertTrue(
            np.allclose(
                [n.start for n in click.notes], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            )
        )
        self.assertEqual(
            [n.pitch for n in click.notes], [37, 31, 31, 31, 37, 31, 31, 31]
        )
        self.assertTrue(
            np.allclose(
                [n.start for n in harmony.notes], [1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2]
            )
        )
        self.assertTrue(
            np.allclose(
                [n.end for n in harmony.notes], [1.5, 1.5, 1.5, 2, 2, 2, 4, 4, 4]
            )
        )
        self.assertTrue(
            np.allclose(
                [n.pitch for n in harmony.notes], [53, 57, 60, 55, 59, 62, 48, 52, 55]
            )
        )
        self.assertTrue(
            np.allclose([n.start for n in melody.notes], [1, 1.25, 1.5, 1.75, 2])
        )
        self.assertTrue(
            np.allclose([n.end for n in melody.notes], [1.125, 1.375, 1.75, 2, 3.0])
        )
        self.assertTrue(
            np.allclose([n.pitch for n in melody.notes], [60, 62, 64, 65, 67])
        )

        with open(
            retrieve_asset("TEST_EXAMPLEANALYSIS_JSON"),
            "r",
            encoding="utf-8-sig",
        ) as f:
            ls = LeadSheet.from_theorytab(json.load(f))
        self.assertEqual(LeadSheet(*ls), ls)
        lily = ls.as_lily()
        lily_expected = r"""
#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        s16*4 d16*12:m g16*20 c16*12 a16*16:m d16*16:m
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 120
        r8 c'8 d'4 e'4 f'4 | g'2 r2~ | r4 g'16 r2~ r8. | e'8 r2~ r4.~ | r1
    }
}

>>

\version "2.18.2"
        """.strip()
        self.assertEqual(lily, lily_expected)

        lily = ls.as_lily(title="Foo", artist="Bar")
        lily_expected = r"""
#(set-default-paper-size "letter")

\header {
    title = "Foo"
    composer = "Bar"
}

<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        s16*4 d16*12:m g16*20 c16*12 a16*16:m d16*16:m
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 120
        r8 c'8 d'4 e'4 f'4 | g'2 r2~ | r4 g'16 r2~ r8. | e'8 r2~ r4.~ | r1
    }
}

>>

\version "2.18.2"
        """.strip()
        self.assertEqual(lily, lily_expected)
