import unittest

from .internal import (
    Chord,
    Harmony,
    Key,
    KeyChanges,
    Melody,
    Meter,
    MeterChanges,
    Note,
    Tempo,
    TempoChanges,
)


class TestInternal(unittest.TestCase):
    def test_meter(self):
        ms = Meter(4, 2, 2)
        self.assertTrue(isinstance(ms, Meter))
        self.assertTrue(isinstance(ms, tuple))
        self.assertEqual(Meter(*ms), ms)
        self.assertEqual(ms, (4, 2, 2))
        self.assertEqual(Meter(4, 2, 2).as_lily(), ("4", "4"))
        self.assertEqual(Meter(3, 2, 2).as_lily(), ("3", "4"))
        with self.assertRaises(NotImplementedError):
            Meter(4, 2, 1)

    def test_tempo(self):
        ts = Tempo(120)
        self.assertTrue(isinstance(ts, Tempo))
        self.assertTrue(isinstance(ts, tuple))
        self.assertEqual(len(ts), 1)
        self.assertEqual(Tempo(*ts), ts)
        ms = Meter(4, 2, 2)
        self.assertEqual(ts.as_lily(ms), ("4", "120"))
        self.assertEqual(
            Tempo(60).as_lily(Meter(3, 2, 2)),
            (
                "4",
                "60",
            ),
        )
        with self.assertRaises(TypeError):
            Tempo(120.5)
        with self.assertRaises(ValueError):
            Tempo(0)
        with self.assertRaises(ValueError):
            Tempo(-1)

    def test_key(self):
        ks = Key(0, (2, 2, 1, 2, 2, 2))
        self.assertTrue(isinstance(ks, Key))
        self.assertTrue(isinstance(ks, tuple))
        self.assertEqual(len(ks), 2)
        self.assertEqual(Key(*ks), ks)
        self.assertEqual(ks.as_lily(), ("c", "major"))
        self.assertEqual(Key(10, (2, 1, 2, 2, 1, 2)).as_lily(), ("bes", "minor"))
        with self.assertRaises(TypeError):
            Key("0", (2, 2, 1, 2, 2, 2))
        with self.assertRaises(TypeError):
            Key(0, "0")
        with self.assertRaises(ValueError):
            Key(0, (0, 2, 2, 1, 2, 2, 2))
        with self.assertRaises(ValueError):
            Key(0, (-1, 2, 2, 1, 2, 2, 2))
        with self.assertRaises(ValueError):
            Key(0, (2, 2, 1, 2, 2, 2, 1))

    def test_note(self):
        ns = Note(0, 0)
        self.assertTrue(isinstance(ns, Note))
        self.assertTrue(isinstance(ns, tuple))
        self.assertEqual(len(ns), 2)
        self.assertEqual(Note(*ns), ns)
        ks = Key(0, (2, 2, 1, 2, 2, 2))
        self.assertEqual(ns.as_lily(ks), ("c", "'"))
        self.assertEqual(
            Note(1, -1).as_lily(Key(11, (2, 2, 1, 2, 2, 2))),
            ("cis", ""),
        )
        with self.assertRaises(TypeError):
            Note("0", 0)
        with self.assertRaises(TypeError):
            Note(0, "0")

    def test_chord(self):
        cs = Chord(0, (4, 3))
        self.assertTrue(isinstance(cs, Chord))
        self.assertTrue(isinstance(cs, tuple))
        self.assertEqual(len(cs), 2)
        self.assertEqual(Chord(*cs), cs)
        ks = Key(0, (2, 2, 1, 2, 2, 2))
        self.assertEqual(cs.as_lily(ks), ("c", ""))
        self.assertEqual(Chord(None, None).as_lily(ks), ("r", ""))
        with self.assertRaises(TypeError):
            Chord("0", (4, 3))
        with self.assertRaises(TypeError):
            Chord(0, "0")
        with self.assertRaises(ValueError):
            Chord(0, (0, 4, 3))
        with self.assertRaises(ValueError):
            Chord(0, (-1, 4, 3))

    def test_meter_changes(self):
        mc = MeterChanges((0, (4, 2, 2)))
        self.assertTrue(isinstance(mc, MeterChanges))
        self.assertTrue(isinstance(mc, tuple))
        self.assertTrue(all(isinstance(k, Meter) for _, k in mc))
        self.assertEqual(len(mc), 1)
        self.assertEqual(MeterChanges(*mc), mc)
        with self.assertRaises(ValueError):
            self.assertEqual(
                MeterChanges(
                    (1, (4, 2, 2)),
                    (0, (4, 2, 2)),
                    (0, (4, 2, 2)),
                ),
                mc,
            )

    def test_tempo_changes(self):
        tc = TempoChanges((0, (120,)))
        self.assertTrue(isinstance(tc, TempoChanges))
        self.assertTrue(isinstance(tc, tuple))
        self.assertTrue(all(isinstance(k, Tempo) for _, k in tc))
        self.assertEqual(len(tc), 1)
        self.assertEqual(TempoChanges(*tc), tc)

    def test_key_changes(self):
        kc = KeyChanges(
            (0, (0, (2, 2, 1, 2, 2, 2))),
        )
        self.assertTrue(isinstance(kc, KeyChanges))
        self.assertTrue(isinstance(kc, tuple))
        self.assertTrue(all(isinstance(k, Key) for _, k in kc))
        self.assertEqual(len(kc), 1)
        self.assertEqual(KeyChanges(*kc), kc)
        self.assertEqual(len(kc), 1)
        self.assertEqual(tuple(sorted(kc)), kc)
        with self.assertRaises(TypeError):
            KeyChanges(
                (0.5, (0, (2, 2, 1, 2, 2, 2))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (-1, (0, (2, 2, 1, 2, 2, 2))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (0, (0, (2, 2, 1, 2, 2, 2, 12))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (0, (0, (2, 2, 1, 2, 2, 2, 12))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (0, (0, (2, 2, 1, 2, 2, 2))),
                (0, (1, (2, 2, 1, 2, 2, 2))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (0, (0, (2, 2, 1, 2, 2, 2))),
                (4, (0, (2, 2, 1, 2, 2, 2))),
            )
        with self.assertRaises(ValueError):
            KeyChanges(
                (0, (0, (2, 2, 1, 2, 2, 2))),
                (0, (0, (2, 2, 1, 2, 2, 2))),
                (1, (0, (2, 2, 1, 2, 2, 2))),
            )

    def test_harmony(self):
        harmony = Harmony(
            (8, (5, (4, 3))),
            (12, (7, (4, 3))),
            (16, (0, (4, 3))),
        )
        self.assertTrue(isinstance(harmony, Harmony))
        self.assertTrue(isinstance(harmony, tuple))
        self.assertTrue(all(isinstance(c, Chord) for _, c in harmony))
        self.assertEqual(len(harmony), 3)
        self.assertEqual(Harmony(*harmony), harmony)

    def test_melody(self):
        melody = Melody(
            (8, 2, (0, 0)),
            (10, 2, (2, 0)),
            (12, 2, (4, 0)),
            (14, 2, (5, 0)),
            (16, 16, (7, 0)),
        )
        self.assertTrue(isinstance(melody, Melody))
        self.assertTrue(isinstance(melody, tuple))
        self.assertTrue(all(isinstance(n, Note) for _, _, n in melody))
        self.assertEqual(len(melody), 5)
        self.assertEqual(Melody(*melody), melody)
