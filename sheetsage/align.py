import numpy as np
from scipy.interpolate import interp1d


def _extrapolating_linear_interp1d(a, b, safe=True):
    if safe:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        if a != sorted(a):
            raise ValueError()
        if b != sorted(b):
            raise ValueError()
        if len(a) != len(b):
            raise ValueError()
        if len(np.unique(a)) != len(a):
            raise ValueError()
        if len(np.unique(b)) != len(b):
            raise ValueError()
    return interp1d(a, b, kind="linear", fill_value="extrapolate")


def create_beat_to_time_fn(beats, times, safe=True):
    return _extrapolating_linear_interp1d(beats, times, safe=safe)


def create_time_to_beat_fn(beats, times, safe=True):
    return _extrapolating_linear_interp1d(times, beats, safe=safe)
