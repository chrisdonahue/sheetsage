import ast
import unittest

from .assets import retrieve_asset
from .infer import Status, sheetsage
from .theory import LeadSheet

# NOTE: Changed legacy output from 177 BPM -> 176 BPM.
_FISHIN_LEGACY_VERBATIM = """
(((0, (4, 2, 2)),), ((0, (176,)),), ((0, (10, (2, 2, 1, 2, 2, 2))),), ((0, (10, (4, 3))), (96, (3, (4, 3))), (128, (10, (4, 3))), (224, (3, (4, 3)))), ((0, 4, (0, 5)), (4, 4, (5, 5)), (8, 4, (5, 5)), (12, 2, (5, 5)), (14, 2, (0, 5)), (16, 2, (7, 5)), (18, 4, (5, 5)), (22, 4, (5, 5)), (26, 2, (2, 5)), (28, 4, (2, 5)), (32, 12, (0, 5)), (44, 12, (10, 4)), (56, 4, (5, 5)), (60, 10, (2, 5)), (70, 4, (3, 5)), (74, 6, (2, 5)), (80, 10, (5, 5)), (90, 38, (2, 5)), (128, 4, (2, 5)), (132, 4, (5, 5)), (136, 4, (5, 5)), (140, 4, (5, 5)), (144, 4, (5, 5)), (148, 4, (5, 5)), (152, 4, (5, 5)), (156, 4, (2, 5)), (160, 4, (0, 5)), (164, 4, (0, 5)), (168, 2, (0, 5)), (170, 2, (10, 4)), (172, 4, (10, 4)), (176, 4, (2, 5)), (180, 1, (5, 5)), (181, 3, (5, 5)), (184, 4, (5, 5)), (188, 4, (2, 5)), (192, 10, (10, 4)), (202, 54, (2, 5)), (256, 23, (7, 5)), (279, 1, (5, 5))), 288)
""".strip()

_FISHIN_REF = """
(((0, (4, 2, 2)),), ((0, (176,)),), ((0, (10, (2, 2, 1, 2, 2, 2))),), ((0, (10, (4, 3))), (96, (3, (4, 3))), (128, (10, (4, 3))), (224, (3, (4, 3)))), ((0, 4, (0, 5)), (4, 4, (5, 5)), (8, 4, (5, 5)), (12, 2, (5, 5)), (14, 2, (0, 5)), (16, 2, (7, 5)), (18, 4, (5, 5)), (22, 4, (5, 5)), (26, 2, (2, 5)), (28, 4, (2, 5)), (32, 12, (0, 5)), (44, 12, (10, 4)), (56, 4, (5, 5)), (60, 14, (2, 5)), (74, 6, (2, 5)), (80, 10, (5, 5)), (90, 38, (2, 5)), (128, 4, (2, 5)), (132, 4, (5, 5)), (136, 2, (5, 5)), (138, 2, (5, 5)), (140, 8, (5, 5)), (148, 4, (5, 5)), (152, 2, (5, 5)), (154, 2, (2, 5)), (156, 4, (2, 5)), (160, 4, (0, 5)), (164, 4, (0, 5)), (168, 2, (0, 5)), (170, 2, (10, 4)), (172, 4, (10, 4)), (176, 4, (2, 5)), (180, 1, (5, 5)), (181, 3, (5, 5)), (184, 4, (5, 5)), (188, 4, (2, 5)), (192, 10, (10, 4)), (202, 54, (2, 5)), (256, 16, (7, 5))), 272)
""".strip()


class TestSheetSage(unittest.TestCase):
    def test_sheetsage(self):
        statuses = []
        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            retrieve_asset("TEST_FISHIN"),
            segment_start_hint=11,
            segment_end_hint=11 + 23.75,
            status_change_callback=lambda s: statuses.append(s),
        )

        self.assertTrue(isinstance(lead_sheet, LeadSheet))
        self.assertTrue(isinstance(segment_beats, list))
        self.assertTrue(isinstance(segment_beats_times, list))
        self.assertEqual(type(segment_beats[0]), int)
        self.assertEqual(type(segment_beats_times[0]), float)

        self.assertEqual(lead_sheet, ast.literal_eval(_FISHIN_REF))
        self.assertEqual(len(segment_beats), len(segment_beats_times))
        self.assertEqual(len(segment_beats), 147)
        self.assertEqual(len(segment_beats_times), 147)
        self.assertEqual(segment_beats[0], -32)
        self.assertEqual(segment_beats[-1], 114)
        self.assertAlmostEqual(segment_beats_times[0], 0.19)
        self.assertAlmostEqual(segment_beats_times[-1], 49.68)

        self.assertEqual(
            statuses,
            [
                Status.DETECTING_BEATS,
                Status.EXTRACTING_FEATURES,
                Status.TRANSCRIBING,
                Status.FORMATTING,
                Status.DONE,
            ],
        )

        # NOTE to future chrisdonahue: To test legacy behavior, need to write out wav file after legacy decode (which uses ffmpeg instead of librosa).

        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            retrieve_asset("TEST_FISHIN"), segment_start_hint=11, legacy_behavior=True
        )

        self.assertEqual(lead_sheet, ast.literal_eval(_FISHIN_LEGACY_VERBATIM))
