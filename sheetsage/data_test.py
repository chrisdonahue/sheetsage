import unittest

from tqdm import tqdm

from .data import iter_hooktheory, load_hooktheory_raw


class TestData(unittest.TestCase):
    def test_hooktheory(self):
        hooktheory_raw = load_hooktheory_raw()
        self.assertEqual(len(hooktheory_raw), 16373)

        hooktheory = list(iter_hooktheory())
        self.assertEqual(len(hooktheory), 16373)
        hooktheory_test = list(iter_hooktheory(split="TEST"))
        self.assertEqual(len(hooktheory_test), 1480)

        hooktheory_raw = load_hooktheory_raw(alignment="USER")
        self.assertEqual(len(hooktheory_raw), 20112)
        hooktheory = list(iter_hooktheory(alignment="USER"))
        self.assertEqual(len(hooktheory), 20112)
