import json
import unittest

import numpy as np

from ..assets import retrieve_asset
from .basic import LilyPitchName, PitchClass
from .internal import Chord, Key, Meter, Note, Tempo
from .theorytab import (
    TheorytabChord,
    TheorytabKey,
    TheorytabMeter,
    TheorytabNote,
    TheorytabTempo,
    TheorytabValueError,
)


class TestTheorytab(unittest.TestCase):
    def test_theorytab_meter(self):
        ttm = TheorytabMeter(
            {
                "beat": 1,
                "numBeats": 4,
                "beatUnit": 1,
            }
        )
        self.assertTrue(isinstance(ttm, TheorytabMeter))
        self.assertTrue(isinstance(ttm, dict))
        self.assertTrue(isinstance(ttm.as_meter(), Meter))
        self.assertEqual(ttm.as_meter(), (4, 2, 2))
        self.assertEqual(
            TheorytabMeter(ttm, numBeats=3, beatUnit=1).as_meter(), (3, 2, 2)
        )
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter()
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter(ttm, extra=None)
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter(ttm, beat=0.5)
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter(ttm, numBeats=1)
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter(ttm, beatUnit=4)
        with self.assertRaises(TheorytabValueError):
            TheorytabMeter(ttm, numBeats=4, beatUnit=3)

    def test_theorytab_tempo(self):
        ttt = TheorytabTempo(
            {
                "beat": 1,
                "bpm": 120,
                "swingFactor": 0,
                "swingBeat": 0.5,
            }
        )
        self.assertTrue(isinstance(ttt, TheorytabTempo))
        self.assertTrue(isinstance(ttt, dict))
        self.assertTrue(isinstance(ttt.as_tempo(), Tempo))
        self.assertEqual(ttt.as_tempo(), (120,))
        self.assertEqual(
            TheorytabTempo(ttt, bpm=60, swingFactor=0.67).as_tempo(), (60,)
        )
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo()
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, extra=None)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, beat=0.5)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, bpm=None)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, bpm=29)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, bpm=301)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, swingFactor=1)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, swingFactor=0.76)
        with self.assertRaises(TheorytabValueError):
            TheorytabTempo(ttt, swingBeat=0.125)

    def test_theorytab_key(self):
        ttk = TheorytabKey({"beat": 1, "scale": "major", "tonic": "C"})
        self.assertTrue(isinstance(ttk, TheorytabKey))
        self.assertTrue(isinstance(ttk, dict))
        self.assertTrue(isinstance(ttk.as_key(), Key))
        self.assertEqual(ttk.as_key(), (0, (2, 2, 1, 2, 2, 2)))
        self.assertEqual(
            TheorytabKey(ttk, scale="minor").as_key(),
            (0, (2, 1, 2, 2, 1, 2)),
        )
        self.assertEqual(
            TheorytabKey(ttk, tonic="Db").as_key(),
            (1, (2, 2, 1, 2, 2, 2)),
        )
        with self.assertRaises(TheorytabValueError):
            TheorytabKey()
        with self.assertRaises(TheorytabValueError):
            TheorytabKey(ttk, extra=None)
        with self.assertRaises(TheorytabValueError):
            TheorytabKey(ttk, beat=0.5)
        with self.assertRaises(TheorytabValueError):
            TheorytabKey(ttk, scale="schmajor")
        with self.assertRaises(TheorytabValueError):
            TheorytabKey(ttk, tonic="Z")

    def test_theorytab_note(self):
        ttk = TheorytabKey({"beat": 1, "scale": "major", "tonic": "C"})
        ttn = TheorytabNote(
            {
                "sd": "1",
                "octave": 0,
                "beat": 1,
                "duration": 1,
                "isRest": False,
                "recordingEndBeat": None,
            }
        )
        self.assertTrue(isinstance(ttn, TheorytabNote))
        self.assertTrue(isinstance(ttn, dict))
        self.assertTrue(isinstance(ttn.as_note(ttk), Note))
        self.assertEqual(TheorytabNote(ttn, sd="bb1").as_note(ttk), (10, -1))
        self.assertEqual(
            TheorytabNote(ttn, sd="bb1").as_note(ttk, legacy_behavior=True),
            (11, -1),
        )
        self.assertEqual(TheorytabNote(ttn, sd="b1").as_note(ttk), (11, -1))
        self.assertEqual(ttn.as_note(ttk), (0, 0))
        self.assertEqual(TheorytabNote(ttn, sd="#1").as_note(ttk), (1, 0))
        self.assertEqual(
            TheorytabNote(ttn, sd="##1").as_note(ttk, legacy_behavior=True),
            (1, 0),
        )
        self.assertEqual(TheorytabNote(ttn, sd="##1").as_note(ttk), (2, 0))
        self.assertEqual(
            TheorytabNote(ttn, sd="3").as_note(TheorytabKey(ttk, tonic="Ab")),
            (0, 1),
        )
        for scale in ["major", "minor"]:
            for pc in range(12):
                for o in range(-3, 3):
                    _ttk = TheorytabKey(
                        ttk, scale=scale, tonic=PitchClass(pc).as_human_pitch_name()
                    )
                    root, intervals = _ttk.as_key()
                    expected = [root] + (root + np.cumsum(intervals)).tolist()
                    expected = (np.array(expected) + (o * 12)).tolist()
                    nss = [
                        TheorytabNote(ttn, sd=str(i), octave=o).as_note(_ttk)
                        for i in range(1, 8)
                    ]
                    self.assertEqual([pc + (o * 12) for pc, o in nss], expected)
        self.assertTrue(ttn.will_sound())
        self.assertFalse(TheorytabNote(ttn, isRest=True).will_sound())
        with self.assertRaises(TheorytabValueError):
            TheorytabNote()
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, extra=None)
        TheorytabNote(ttn, beat=0, isRest=True)
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, beat=0)
        TheorytabNote(ttn, duration=0, isRest=True)
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, duration=0)
        TheorytabNote(ttn, duration=1e-7)
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, duration=1e-9)
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, sd="bbb1")
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, sd="a1")
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, sd="8")
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, octave=-5)
        with self.assertRaises(TheorytabValueError):
            TheorytabNote(ttn, octave=5)

    def test_theorytab_chord(self):
        ttk = TheorytabKey({"beat": 1, "scale": "major", "tonic": "C"})
        ttc = TheorytabChord(
            {
                "root": 1,
                "beat": 1,
                "duration": 4,
                "type": 5,
                "inversion": 0,
                "applied": 0,
                "adds": [],
                "omits": [],
                "alterations": [],
                "suspensions": [],
                "pedal": None,
                "alternate": "",
                "borrowed": "",
                "isRest": False,
                "recordingEndBeat": None,
            }
        )
        self.assertTrue(isinstance(ttc, TheorytabChord))
        self.assertTrue(isinstance(ttc, dict))
        self.assertTrue(isinstance(ttc.as_chord(ttk), Chord))
        self.assertEqual(ttc.as_chord(ttk), (0, (4, 3)))
        self.assertEqual(ttc.as_chord(TheorytabKey(ttk, scale="minor")), (0, (3, 4)))
        self.assertEqual(TheorytabChord(ttc, root=4).as_chord(ttk), (5, (4, 3)))
        self.assertEqual(
            [
                TheorytabChord(ttc, root=i).as_chord(ttk).as_lily(ttk.as_key())
                for i in range(1, 8)
            ],
            [
                ("c", ""),
                ("d", "m"),
                ("e", "m"),
                ("f", ""),
                ("g", ""),
                ("a", "m"),
                ("b", "dim"),
            ],
        )

    def test_theorytab_convert_to_ly(self):
        ref_ly = """
        c4*4  d4*4:m  d4*4:m7  g4*4:7  c4*4:maj7  c4*4:sus4  c4*4:9^7  b4*4:dim  b4*4:m7.5-  c4*4:sus2  g4*4:7sus4  d4*4:m9  c4*4:maj9  g4*4:7sus2  d4*4:m9^7  b4*4:dim7  c4*4:maj7sus2  g4*4:9  c4*4:11.9^7  c4*4:aug  
        """.strip()
        ref = []
        for i, c in enumerate(ref_ly.split()):
            ly_lypn = c[0]
            if ":" in c:
                ly_chord_name = c.split(":")[1]
            else:
                assert i == 0
                ly_chord_name = ""
            ref.append((LilyPitchName(ly_lypn), ly_chord_name))
        assert len(set([n for _, n in ref])) == len(ref)

        with open(
            retrieve_asset("TEST_COMMONCHORDS_JSON"),
            "r",
            encoding="utf-8-sig",
        ) as f:
            analysis = json.load(f)
        key = TheorytabKey(analysis["keys"][0]).as_key()
        chords = [TheorytabChord(c) for c in analysis["chords"]]
        parsed = [c.as_chord(key).as_lily(key) for c in chords]
        self.assertEqual(parsed, ref)
