import unittest

from .internal import Harmony, KeyChanges, Melody, MeterChanges
from .utils import estimate_key_changes, theorytab_find_applicable


class TestUtils(unittest.TestCase):
    def test_theorytab_find_applicable(self):
        timed_events = [{"beat": 1}, {"beat": 17}, {"beat": 33}]
        self.assertEqual(
            theorytab_find_applicable(timed_events, {"beat": 1}), {"beat": 1}
        )
        self.assertEqual(
            theorytab_find_applicable(timed_events, {"beat": 16.9}), {"beat": 1}
        )
        self.assertEqual(
            theorytab_find_applicable(timed_events, {"beat": 16.9999}), {"beat": 17}
        )
        self.assertEqual(
            theorytab_find_applicable(timed_events, {"beat": 17}), {"beat": 17}
        )
        self.assertEqual(
            theorytab_find_applicable(timed_events, {"beat": 100000}), {"beat": 33}
        )
        with self.assertRaises(ValueError):
            theorytab_find_applicable(timed_events, {"beat": 0.5})

    def test_estimate_key_changes(self):
        meter_changes = MeterChanges((0, (4, 2, 2)))
        # C major
        self.assertEqual(
            estimate_key_changes(
                meter_changes,
                Harmony(
                    (8, (5, (4, 3))),
                    (12, (7, (4, 3))),
                    (16, (0, (4, 3))),
                ),
                Melody(
                    (8, 2, (0, 0)),
                    (10, 2, (2, 0)),
                    (12, 2, (4, 0)),
                    (14, 2, (5, 0)),
                    (16, 8, (7, 0)),
                ),
            ),
            KeyChanges((0, (0, (2, 2, 1, 2, 2, 2)))),
        )
        # G minor
        self.assertEqual(
            estimate_key_changes(
                meter_changes,
                Harmony(
                    (8, (0, (3, 4))),
                    (12, (2, (3, 4))),
                    (16, (7, (3, 4))),
                ),
                Melody(
                    (8, 2, (7, 0)),
                    (10, 2, (9, 0)),
                    (12, 2, (10, 0)),
                    (14, 2, (0, 1)),
                    (16, 8, (2, 1)),
                ),
            ),
            KeyChanges((0, (7, (2, 1, 2, 2, 1, 2)))),
        )
        # Bb major
        self.assertEqual(
            estimate_key_changes(
                meter_changes,
                Harmony(
                    (8, (3, (4, 3))),
                    (12, (5, (4, 3))),
                    (16, (10, (4, 3))),
                ),
                Melody(
                    (8, 2, (10, 0)),
                    (10, 2, (0, 0)),
                    (12, 2, (2, 0)),
                    (14, 2, (3, 0)),
                    (16, 8, (5, 0)),
                ),
            ),
            KeyChanges((0, (10, (2, 2, 1, 2, 2, 2)))),
        )
        with self.assertRaises(Exception):
            estimate_key_changes(meter_changes, Harmony([]), Melody([]))
