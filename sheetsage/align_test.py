import unittest

import numpy as np

from .align import create_beat_to_time_fn, create_time_to_beat_fn


class Test(unittest.TestCase):
    def test_beat_time_conversion(self):
        beats = [0, 1, 2, 3]
        times = [1.0, 2.0, 2.5, 3.0]
        eps = 1e-6
        test_beats = np.arange(-1, 4 + 1e-6, 0.5)
        test_times = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.50]
        )

        beat_to_time_fn = create_beat_to_time_fn(beats, times)
        self.assertEqual(beat_to_time_fn(0), 1.0)
        self.assertTrue(np.array_equal(beat_to_time_fn(test_beats), test_times))

        time_to_beat_fn = create_time_to_beat_fn(beats, times)
        self.assertEqual(time_to_beat_fn(1.0), 0.0)
        self.assertTrue(np.array_equal(time_to_beat_fn(test_times), test_beats))
