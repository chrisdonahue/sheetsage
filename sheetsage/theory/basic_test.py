import unittest

from .basic import HumanPitchName, LilyPitchName, PitchClass, PitchInterval


class TestBasic(unittest.TestCase):
    def test_pitch_class(self):
        pc = PitchClass(0)
        self.assertTrue(isinstance(pc, PitchClass))
        self.assertTrue(isinstance(pc, int))
        pcs = [PitchClass(i) for i in range(12)]
        self.assertEqual(
            " ".join([pc.as_human_pitch_name(enharmonics="b") for pc in pcs]),
            "C Db D Eb E F Gb G Ab A Bb B",
        )
        self.assertEqual(
            " ".join([pc.as_human_pitch_name(enharmonics="#") for pc in pcs]),
            "C C# D D# E F F# G G# A A# B",
        )
        self.assertEqual(
            " ".join([pc.as_lily_pitch_name(enharmonics="es") for pc in pcs]),
            "c des d ees e f ges g aes a bes b",
        )
        self.assertEqual(
            " ".join([pc.as_lily_pitch_name(enharmonics="is") for pc in pcs]),
            "c cis d dis e f fis g gis a ais b",
        )
        with self.assertRaises(TypeError):
            PitchClass("0")
        with self.assertRaises(ValueError):
            PitchClass(-1)
        with self.assertRaises(ValueError):
            PitchClass(12)

    def test_pitch_interval(self):
        pi = PitchInterval(0)
        self.assertTrue(isinstance(pi, PitchInterval))
        self.assertTrue(isinstance(pi, int))
        pis = [PitchInterval(i) for i in range(-12, 13)]
        with self.assertRaises(TypeError):
            PitchInterval("0")

    def test_human_pitch_name(self):
        hrpn = HumanPitchName("C")
        self.assertTrue(isinstance(hrpn, HumanPitchName))
        self.assertTrue(isinstance(hrpn, str))
        self.assertEqual(
            [
                HumanPitchName(hrpn).as_pitch_class()
                for hrpn in "C Db D Eb E F Gb G Ab A Bb B".split()
            ],
            list(range(12)),
        )
        self.assertEqual(
            [
                HumanPitchName(hrpn).as_pitch_class()
                for hrpn in "C C# D D# E F F# G G# A A# B".split()
            ],
            list(range(12)),
        )
        self.assertEqual(
            [
                HumanPitchName(hrpn).as_lily_pitch_name()
                for hrpn in "C Db D Eb E F Gb G Ab A Bb B".split()
            ],
            "c des d ees e f ges g aes a bes b".split(),
        )
        self.assertEqual(
            [
                HumanPitchName(hrpn).as_lily_pitch_name()
                for hrpn in "C C# D D# E F F# G G# A A# B".split()
            ],
            "c cis d dis e f fis g gis a ais b".split(),
        )
        self.assertEqual(HumanPitchName("Cbb").as_pitch_class(), 10)
        self.assertEqual(HumanPitchName("C##").as_pitch_class(), 2)
        with self.assertRaises(TypeError):
            HumanPitchName(0)
        with self.assertRaises(ValueError):
            HumanPitchName("")
        with self.assertRaises(ValueError):
            HumanPitchName("a")
        with self.assertRaises(ValueError):
            HumanPitchName("Z")
        with self.assertRaises(ValueError):
            HumanPitchName("CB")
        with self.assertRaises(ValueError):
            HumanPitchName("Cn")
        with self.assertRaises(ValueError):
            HumanPitchName("Cb#")
        with self.assertRaises(ValueError):
            HumanPitchName("Cbbb")
        with self.assertRaises(ValueError):
            HumanPitchName("C###")

    def test_lily_pitch_name(self):
        lypn = LilyPitchName("c")
        self.assertTrue(isinstance(lypn, LilyPitchName))
        self.assertTrue(isinstance(lypn, str))
        self.assertEqual(
            [
                LilyPitchName(lypn).as_pitch_class()
                for lypn in "c des d ees e f ges g aes a bes b".split()
            ],
            list(range(12)),
        )
        self.assertEqual(
            [
                LilyPitchName(lypn).as_pitch_class()
                for lypn in "c cis d dis e f fis g gis a ais b".split()
            ],
            list(range(12)),
        )
        self.assertEqual(
            [
                LilyPitchName(lypn).as_human_pitch_name()
                for lypn in "c des d ees e f ges g aes a bes b".split()
            ],
            "C Db D Eb E F Gb G Ab A Bb B".split(),
        )
        self.assertEqual(
            [
                LilyPitchName(lypn).as_human_pitch_name()
                for lypn in "c cis d dis e f fis g gis a ais b".split()
            ],
            "C C# D D# E F F# G G# A A# B".split(),
        )
        self.assertEqual(LilyPitchName("ceses").as_pitch_class(), 10)
        self.assertEqual(LilyPitchName("cisis").as_pitch_class(), 2)
        with self.assertRaises(TypeError):
            LilyPitchName(0)
        with self.assertRaises(ValueError):
            LilyPitchName("")
        with self.assertRaises(ValueError):
            LilyPitchName("A")
        with self.assertRaises(ValueError):
            LilyPitchName("z")
        with self.assertRaises(ValueError):
            LilyPitchName("cb")
        with self.assertRaises(ValueError):
            LilyPitchName("cas")
        with self.assertRaises(ValueError):
            LilyPitchName("cesis")
        with self.assertRaises(ValueError):
            LilyPitchName("ceseseses")
        with self.assertRaises(ValueError):
            LilyPitchName("cisisisis")
