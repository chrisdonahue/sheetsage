import numpy as np

from .basic import PitchClass, PitchInterval

_LILY_NAME_TO_SCALE_DEGREES = {
    "major": (2, 2, 1, 2, 2, 2),
    "dorian": (2, 1, 2, 2, 2, 1),
    "phrygian": (1, 2, 2, 2, 1, 2),
    "lydian": (2, 2, 2, 1, 2, 2),
    "mixolydian": (2, 2, 1, 2, 2, 1),
    "minor": (2, 1, 2, 2, 1, 2),
    "locrian": (1, 2, 2, 1, 2, 2),
}

_SCALE_DEGREES_TO_LILY_NAME = {v: k for k, v in _LILY_NAME_TO_SCALE_DEGREES.items()}


_KEY_TO_ENHARMONICS = {}
for o, ln in zip(
    [0, 2, 4, 5, 7, 9, 11],
    ["major", "dorian", "phrygian", "lydian", "mixolydian", "minor", "locrian"],
):
    for i, pc in enumerate(range(12)):
        pc = (o + (pc * 7)) % 12
        ks = (pc, _LILY_NAME_TO_SCALE_DEGREES[ln])
        enharmonics = "#" if i <= 6 else "b"
        _KEY_TO_ENHARMONICS[ks] = enharmonics


_CHORD_DEGREES_TO_LILY_NAME = {
    (4, 3): "",  # Major
    (3, 4): "m",  # Minor
    (3, 4, 3): "m7",  # Minor 7
    (4, 3, 3): "7",  # 7
    (4, 3, 4): "maj7",  # Major 7
    (5, 2): "sus4",  # Sus 4
    (4, 3, 7): "9^7",  # Add 9
    (3, 3): "dim",  # Dim
    (3, 3, 4): "m7.5-",  # Dim minor 7
    (2, 5): "sus2",  # Sus 2
    (5, 2, 3): "7sus4",  # 7 Sus 4
    (3, 4, 3, 4): "m9",  # Minor 9
    (4, 3, 4, 3): "maj9",  # Major 9
    (2, 5, 3): "7sus2",  # 7 Sus 2
    (3, 4, 7): "m9^7",  # Minor Add 9
    (3, 3, 3): "dim7",  # Dim 7
    (2, 5, 4): "maj7sus2",  # Major 7 Sus 2
    (4, 3, 3, 4): "9",  # 9
    (2, 3, 2): "11.9^7",  # Sus 2 Sus 4
    (4, 4): "aug",  # Aug
}

_LILY_ABSOLUTE_OCTAVE = 3


class _ImmutableIterable(tuple):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class Meter(_ImmutableIterable):
    # Semantics: https://en.wikipedia.org/wiki/Metre_(music)#Metric_structure
    # Meter is defined by "pulse groups": collections of repeating patterns of accented pulses, i.e., measures
    # A "pulse group" has a "primary" and "secondary" subdivision
    # "Primary subdivision" is synonymous with "pulse"
    # Primary subdivision names: "duple" = 2, "triple" = 3, "quadruple" = 4, ...
    # Secondary subdivision names: "simple" = 2, "compound" = 3
    # Here we add a tertiary subdivision which refers to the quantization
    # Time signatures are abstractions of meter which also take readability into account
    # E.g., 3/4 with sixteenth note subdivisions is (3, 2, 2), because we have a simple triple meter w/ each secondary subdivision divided twice
    def __new__(cls, primary, secondary, tertiary):
        meter = (primary, secondary, tertiary)
        if meter not in [(3, 2, 2), (4, 2, 2)]:
            raise NotImplementedError()
        return super().__new__(cls, meter)

    def as_lily(self):
        assert self in [(3, 2, 2), (4, 2, 2)]
        if self == (3, 2, 2):
            result = ("3", "4")
        else:
            result = ("4", "4")
        return result


class Tempo(_ImmutableIterable):
    def __new__(cls, pulses_per_minute):
        if not isinstance(pulses_per_minute, int):
            raise TypeError()
        if pulses_per_minute <= 0:
            raise ValueError()
        return super().__new__(cls, (pulses_per_minute,))

    def as_lily(self, meter):
        if not isinstance(meter, Meter):
            raise TypeError()
        assert meter in [(3, 2, 2), (4, 2, 2)]
        return ("4", str(self[0]))


class Key(_ImmutableIterable):
    def __new__(cls, root_pc, scale_degree_pis):
        if not isinstance(root_pc, int):
            raise TypeError()
        if not isinstance(scale_degree_pis, tuple):
            raise TypeError()
        for pi in scale_degree_pis:
            if pi <= 0:
                raise ValueError()
        if sum(scale_degree_pis) >= 12:
            raise ValueError()
        return super().__new__(
            cls,
            (PitchClass(root_pc), tuple(PitchInterval(pi) for pi in scale_degree_pis)),
        )

    def as_lily(self):
        enharmonics = _KEY_TO_ENHARMONICS.get(self)
        if enharmonics is None:
            raise ValueError("Unknown scale")
        root = self[0].as_human_pitch_name(enharmonics=enharmonics).as_lily_pitch_name()
        scale_name = _SCALE_DEGREES_TO_LILY_NAME.get(self[1])
        assert scale_name is not None
        return (root, scale_name)


class Note(_ImmutableIterable):
    def __new__(cls, pc, octave):
        if not isinstance(pc, int):
            raise TypeError()
        if not isinstance(octave, int):
            raise TypeError()
        return super().__new__(cls, (PitchClass(pc), int(octave)))

    def as_lily(self, key, octave_0=4):
        if not isinstance(key, Key):
            raise TypeError()
        enharmonics = _KEY_TO_ENHARMONICS.get(key)
        if enharmonics is None:
            raise ValueError()
        pitch_class_str = (
            self[0].as_human_pitch_name(enharmonics=enharmonics).as_lily_pitch_name()
        )
        octave = (octave_0 - _LILY_ABSOLUTE_OCTAVE) + self[1]
        octave_str = ("'" if octave > 0 else ",") * abs(octave)
        return (pitch_class_str, octave_str)

    def as_midi_pitch(self, octave_0=4):
        return 12 + (12 * (octave_0 + self[1])) + self[0]


class Chord(_ImmutableIterable):
    def __new__(cls, root_pc, chord_degree_pis):
        if root_pc is None:
            if chord_degree_pis is not None:
                raise ValueError()
            t = (None, None)
        else:
            if not isinstance(root_pc, int):
                raise TypeError()
            if not isinstance(chord_degree_pis, tuple):
                raise TypeError()
            for pi in chord_degree_pis:
                if pi <= 0:
                    raise ValueError()
            t = (
                PitchClass(root_pc),
                tuple(PitchInterval(pi) for pi in chord_degree_pis),
            )
        return super().__new__(cls, t)

    def as_lily(self, key):
        if not isinstance(key, Key):
            raise TypeError()
        if self[0] is None:
            assert self[1] is None
            result = ("r", "")
        else:
            enharmonics = _KEY_TO_ENHARMONICS.get(key)
            if enharmonics is None:
                raise ValueError()
            root_str = (
                self[0]
                .as_human_pitch_name(enharmonics=enharmonics)
                .as_lily_pitch_name()
            )
            result = (root_str, _CHORD_DEGREES_TO_LILY_NAME.get(self[1], self[1]))
        return result

    def as_midi_pitches(self, octave_0=3):
        if self[0] is None:
            assert self[1] is None
            result = []
        else:
            chord = [self[0]] + (self[0] + np.cumsum(self[1])).tolist()
            result = (12 + (12 * octave_0) + np.array(chord)).tolist()
        return result


class _InstantEvent(_ImmutableIterable):
    def __new__(cls, event_cls, onset, event):
        if not isinstance(onset, int):
            raise TypeError()
        if onset < 0:
            raise ValueError()
        event = event_cls(*event)
        assert isinstance(onset, int)
        assert isinstance(event, event_cls)
        assert isinstance(event, _ImmutableIterable)
        return super().__new__(cls, (onset, event))


class _SustainedEvent(_ImmutableIterable):
    def __new__(cls, event_cls, onset, duration, event):
        onset, event = _InstantEvent(event_cls, onset, event)
        if not isinstance(duration, int):
            raise TypeError()
        if duration < 0:
            raise ValueError()
        assert isinstance(onset, int)
        assert isinstance(duration, int)
        assert isinstance(event, event_cls)
        assert isinstance(event, _ImmutableIterable)
        return super().__new__(cls, (onset, duration, event))


class _InstantEventList(_ImmutableIterable):
    def __new__(cls, event_cls, *args):
        # Instantiate and sort
        instant_events = [_InstantEvent(event_cls, *ie) for ie in args]
        instant_events = sorted(instant_events, key=lambda ie: ie[0])

        # Check that no two start on same offset
        offsets = set(ie[0] for ie in instant_events)
        if len(offsets) != len(instant_events):
            raise ValueError()

        # Check that all events constitute a change
        last_ie = None
        for ie in instant_events:
            if last_ie is not None and ie[1] == last_ie[1]:
                raise ValueError()
            last_ie = ie

        return super().__new__(cls, tuple(instant_events))


class _SustainedEventList(_ImmutableIterable):
    def __new__(cls, event_cls, *args):
        # Instantiate and sort
        sustained_events = [_SustainedEvent(event_cls, *se) for se in args]
        sustained_events = sorted(sustained_events, key=lambda se: se[0])

        # Check that no two start on same offset
        offsets = set(se[0] for se in sustained_events)
        if len(offsets) != len(sustained_events):
            raise ValueError()

        # Check that all durations are nonzero
        if any(d == 0 for _, d, _ in sustained_events):
            raise ValueError()

        # Ensure monophonic
        for i, se in enumerate(sustained_events):
            if i + 1 < len(sustained_events):
                max_duration = sustained_events[i + 1][0] - se[0]
                assert max_duration > 0
                if se[1] > max_duration:
                    raise ValueError()

        return super().__new__(cls, tuple(sustained_events))


class _DefinedAtStartInstantEventList(_InstantEventList):
    def __new__(cls, event_cls, *args):
        instant_events = super().__new__(cls, event_cls, *args)
        if len(instant_events) == 0 or instant_events[0][0] != 0:
            raise ValueError()
        return instant_events


class MeterChanges(_DefinedAtStartInstantEventList):
    def __new__(cls, *args):
        return super().__new__(cls, Meter, *args)


class TempoChanges(_DefinedAtStartInstantEventList):
    def __new__(cls, *args):
        return super().__new__(cls, Tempo, *args)


class KeyChanges(_DefinedAtStartInstantEventList):
    def __new__(cls, *args):
        return super().__new__(cls, Key, *args)


class Harmony(_InstantEventList):
    def __new__(cls, *args):
        return super().__new__(cls, Chord, *args)


class Melody(_SustainedEventList):
    def __new__(cls, *args):
        return super().__new__(cls, Note, *args)
