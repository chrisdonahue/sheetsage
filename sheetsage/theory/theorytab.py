import copy
from collections import OrderedDict

import numpy as np

from .basic import HumanPitchName
from .internal import Chord, Key, Meter, Note, Tempo

_THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS = {
    "major": (2, 2, 1, 2, 2, 2),
    "dorian": (2, 1, 2, 2, 2, 1),
    "phrygian": (1, 2, 2, 2, 1, 2),
    "lydian": (2, 2, 2, 1, 2, 2),
    "mixolydian": (2, 2, 1, 2, 2, 1),
    "minor": (2, 1, 2, 2, 1, 2),
    "locrian": (1, 2, 2, 1, 2, 2),
    "harmonicMinor": (2, 1, 2, 2, 1, 3),
    "phrygianDominant": (1, 3, 1, 2, 1, 2),
}

_THEORYTAB_ACCIDENTAL_STR_TO_NUM_SEMITONES = {"bb": -2, "b": -1, "": 0, "#": 1, "##": 2}
_THEORYTAB_ACCIDENTAL_STR_TO_NUM_SEMITONES_LEGACY_BUG = {
    "bb": -1,
    "b": -1,
    "": 0,
    "#": 1,
    "##": 1,
}

# NOTE: Rules written down manually from Hookpad
_THEORYTAB_CHORD_TYPE_TO_ALLOWED_OPTIONS = {
    5: {
        "inversions": [0, 1, 2],
        "suspensions": [2, 4],
        "adds": [9, 4, 6],
        "omits": [3, 5],
        "alterations": ["b5", "#5"],
    },
    7: {
        "inversions": [0, 1, 2, 3],
        "suspensions": [2, 4],
        "adds": [4, 6],
        "omits": [3, 5],
        "alterations": ["b5", "#5", "b9", "#9", "#11", "b13"],
    },
    9: {
        "inversions": [0],
        "suspensions": [4],
        "adds": [6],
        "omits": [3, 5],
        "alterations": ["b5", "#5", "#11", "b13"],
    },
    11: {
        "inversions": [0],
        "suspensions": [2],
        "adds": [],
        "omits": [3, 5],
        "alterations": ["b5", "#5", "b9", "#9", "b13"],
    },
    13: {
        "inversions": [0],
        "suspensions": [],
        "adds": [],
        "omits": [3, 5],
        "alterations": ["b5", "#5", "b9", "#9", "#11"],
    },
}


class _TheorytabDict(dict):
    _FIELDS = []

    def _check_values(self):
        if "beat" in self:
            min_beat = 0 if "isRest" in self and self["isRest"] else 1
            if self["beat"] < min_beat:
                raise TheorytabValueError("beat")

    def __init__(self, *args, _skip_value_checks=False, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self._FIELDS:
            if f not in self:
                raise TheorytabValueError(f"Missing field '{f}'")
        for f in self.keys():
            if f not in self._FIELDS:
                raise TheorytabValueError(f"Unknown field '{f}'")
        if not _skip_value_checks:
            self._check_values()


class TheorytabValueError(ValueError):
    pass


class TheorytabMeter(_TheorytabDict):
    _FIELDS = ["beat", "numBeats", "beatUnit"]

    def _check_values(self):
        super()._check_values()
        if self["numBeats"] not in [2, 3, 4, 5, 6, 9, 12]:
            raise TheorytabValueError("numBeats")
        if self["beatUnit"] not in [1, 3]:
            raise TheorytabValueError("beatUnit")
        if self["beatUnit"] == 3 and self["numBeats"] not in [3, 6, 9, 12]:
            raise TheorytabValueError("numBeats,beatUnit")

    def as_meter(self):
        assert self["numBeats"] % self["beatUnit"] == 0
        return Meter(
            self["numBeats"] // self["beatUnit"],
            3 if self["beatUnit"] == 3 else 2,
            4 if self["beatUnit"] == 3 else 2,
        )


class TheorytabTempo(_TheorytabDict):
    _FIELDS = ["beat", "bpm", "swingFactor", "swingBeat"]

    def _check_values(self):
        super()._check_values()
        if self["bpm"] is None or self["bpm"] < 30 or self["bpm"] > 300:
            raise TheorytabValueError("bpm")
        if isinstance(self["swingFactor"], int):
            if self["swingFactor"] != 0:
                raise TheorytabValueError("swingFactor")
        else:
            if self["swingFactor"] < 0.5 or self["swingFactor"] > 0.75:
                raise TheorytabValueError("swingFactor")
        if self["swingBeat"] not in [0.5, 0.25]:
            raise TheorytabValueError("swingBeat")

    def as_tempo(self):
        return Tempo(round(self["bpm"]))


class TheorytabKey(_TheorytabDict):
    _FIELDS = ["beat", "scale", "tonic"]

    def _check_values(self):
        super()._check_values()
        if self["scale"] not in _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS:
            raise TheorytabValueError("scale")
        try:
            HumanPitchName(self["tonic"])
        except ValueError:
            raise TheorytabValueError("tonic")

    def as_key(self):
        return Key(
            HumanPitchName(self["tonic"]).as_pitch_class(),
            _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[self["scale"]],
        )


class _TheorytabNoteOrChord(_TheorytabDict):
    def _check_values(self, eps=1e-8):
        super()._check_values()
        assert "isRest" in self
        assert "duration" in self
        min_duration = None if self["isRest"] else eps
        if min_duration is not None and self["duration"] < min_duration:
            raise TheorytabValueError("duration")

    def will_sound(self, min_duration=1e-8):
        return (
            self["beat"] >= 1 and self["duration"] > min_duration and not self["isRest"]
        )


class TheorytabNote(_TheorytabNoteOrChord):
    _FIELDS = ["sd", "octave", "beat", "duration", "isRest", "recordingEndBeat"]

    def _check_values(self):
        super()._check_values()
        sd = self["sd"]
        if len(sd) not in [1, 2, 3]:
            raise TheorytabValueError("sd")
        accidental_str = sd[:-1]
        if accidental_str not in _THEORYTAB_ACCIDENTAL_STR_TO_NUM_SEMITONES:
            raise TheorytabValueError("sd")
        sd = int(sd[-1]) - 1
        if sd < 0 or sd >= 7:
            raise TheorytabValueError("sd")
        if abs(self["octave"]) > 4:
            raise TheorytabValueError("octave")

    def as_note(self, key, legacy_behavior=False):
        if isinstance(key, dict) and not isinstance(key, TheorytabKey):
            key = TheorytabKey(key)
        if isinstance(key, TheorytabKey):
            key = key.as_key()
        if not isinstance(key, Key):
            raise TypeError()
        result = None
        if self.will_sound(**({"min_duration": 0} if legacy_behavior else {})):
            key_tonic_pc, key_scale_intervals = key
            sd = self["sd"]
            accidental_str = sd[:-1]
            if legacy_behavior:
                accidental = _THEORYTAB_ACCIDENTAL_STR_TO_NUM_SEMITONES_LEGACY_BUG[
                    accidental_str
                ]
            else:
                accidental = _THEORYTAB_ACCIDENTAL_STR_TO_NUM_SEMITONES[accidental_str]
            sd = int(sd[-1]) - 1
            pitch = 12 * self["octave"]
            pitch += key_tonic_pc
            pitch += sum(key_scale_intervals[:sd])
            pitch += accidental
            result = Note(pitch % 12, pitch // 12)
        return result


class TheorytabChord(_TheorytabNoteOrChord):
    _FIELDS = [
        "root",
        "beat",
        "duration",
        "type",
        "inversion",
        "applied",
        "adds",
        "omits",
        "alterations",
        "suspensions",
        "pedal",
        "alternate",
        "borrowed",
        "isRest",
        "recordingEndBeat",
    ]

    def _check_values(self):
        super()._check_values()

        # Check 'root'
        if self["root"] <= 0:
            if self.will_sound():
                raise TheorytabValueError("root")
        else:
            if self["root"] not in [1, 2, 3, 4, 5, 6, 7]:
                raise TheorytabValueError("root")

        # Check 'type'
        if self["type"] not in [5, 7, 9, 11, 13]:
            raise TheorytabValueError("type")

        # Check 'inversion'
        if self["inversion"] not in [0, 1, 2, 3]:
            raise TheorytabValueError("inversion")

        # Check 'applied'
        if self["applied"] not in [0, 1, 2, 3, 4, 5, 6, 7]:
            raise TheorytabValueError("applied")

        # Check 'adds'
        if any(a not in [9, 4, 6] for a in self["adds"]):
            raise TheorytabValueError("adds")
        if len(self["adds"]) != len(set(self["adds"])):
            raise TheorytabValueError("adds")

        # Check 'omits'
        if any(o not in [3, 5] for o in self["omits"]):
            raise TheorytabValueError("omits")
        if len(self["omits"]) != len(set(self["omits"])):
            raise TheorytabValueError("omits")

        # Check 'alterations'
        if any(
            a not in ["b5", "#5", "b9", "#9", "#11", "b13"] for a in self["alterations"]
        ):
            raise TheorytabValueError("alterations")
        if len(self["alterations"]) != len(set(self["alterations"])):
            raise TheorytabValueError("alterations")

        # Check 'suspensions'
        if any(s not in [2, 4] for s in self["suspensions"]):
            raise TheorytabValueError("suspensions")
        if len(self["suspensions"]) != len(set(self["suspensions"])):
            raise TheorytabValueError("suspensions")

        # Check 'pedal'
        if self["pedal"] is not None:
            raise TheorytabValueError("pedal")

        # Check 'alternate'
        if self["alternate"] != "":
            raise TheorytabValueError("alternate")

        # Check 'borrowed'
        if not (
            self["borrowed"] == ""
            or self["borrowed"] is None
            or (isinstance(self["borrowed"], list) and len(self["borrowed"]) == 7)
            or self["borrowed"] in _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS
        ):
            raise TheorytabValueError("borrowed")

        # Check 'type' against others
        for name, allowed_options in _THEORYTAB_CHORD_TYPE_TO_ALLOWED_OPTIONS[
            self["type"]
        ].items():
            if name == "inversions":
                if self["inversion"] not in allowed_options:
                    raise TheorytabValueError("type,inversion")
            else:
                if any(o not in allowed_options for o in self[name]):
                    raise TheorytabValueError(f"type,{name}")

        # Check 'inversion,omits'
        if self["inversion"] == 1 and 3 in self["omits"]:
            raise TheorytabValueError("inversion,omits")
        if self["inversion"] == 2 and 5 in self["omits"]:
            raise TheorytabValueError("inversion,omits")

        # Check 'adds,alterations'
        for alt in self["alterations"]:
            if int(alt[-1]) in self["adds"]:
                raise TheorytabValueError("adds,alterations")

        # Check 'adds,suspensions'
        if 2 in self["suspensions"] and 9 in self["adds"]:
            raise TheorytabValueError("adds,suspensions")
        if 4 in self["suspensions"] and 4 in self["adds"]:
            raise TheorytabValueError("adds,suspensions")

        # Check 'omits,alterations'
        if 5 in self["omits"] and (
            "b5" in self["alterations"] or "#5" in self["alterations"]
        ):
            raise TheorytabValueError("omits,alterations")

        # Check 'omits,suspensions'
        if 3 in self["omits"] and len(self["suspensions"]) > 0:
            raise TheorytabValueError("omits,suspensions")

    def as_chord(self, key, root_position=False):
        if isinstance(key, dict) and not isinstance(key, TheorytabKey):
            key = TheorytabKey(key)
        if isinstance(key, TheorytabKey):
            key = key.as_key()
        if not isinstance(key, Key):
            raise TypeError()
        result = None
        if self.will_sound():
            chord = copy.deepcopy(self)

            # Unsupported
            # TODO:
            if not root_position and chord["inversion"] != 0:
                raise NotImplementedError("inversion")

            # Build chord scale degrees
            chord_degrees = set(range(1, chord["type"] + 1, 2))

            # Apply suspensions
            for i, d in enumerate(chord["suspensions"]):
                if i == 0:
                    assert 3 in chord_degrees
                    chord_degrees.remove(3)
                assert d not in chord_degrees
                chord_degrees.add(d)

            # Apply adds
            for d in chord["adds"]:
                if d in [4, 6]:
                    d += 7
                chord_degrees.add(d)

            # Apply omits
            for d in chord["omits"]:
                assert d in [3, 5]
                assert d in chord_degrees
                chord_degrees.remove(d)

            # Apply alterations
            for d in chord["alterations"]:
                d = int(d[1:])
                chord_degrees.add(d)

            # Convert to list
            chord_degrees = sorted(list(chord_degrees))

            # Find scale intervals
            key_tonic_pc, key_scale_intervals = key

            # Apply borrow (changes intervals)
            if isinstance(chord["borrowed"], list):
                key_scale_intervals = chord["borrowed"]
            else:
                if chord["borrowed"] in _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS:
                    key_scale_intervals = _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[
                        chord["borrowed"]
                    ]
                key_scale_intervals = [0] + np.cumsum(key_scale_intervals).tolist()
            assert len(key_scale_intervals) == 7

            # Apply secondary (changes tonic and intervals)
            major_scale_intervals = _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS["major"]
            major_scale_intervals = [0] + np.cumsum(major_scale_intervals).tolist()
            if chord["applied"] > 0:
                key_tonic_pc = (
                    key_tonic_pc + key_scale_intervals[chord["root"] - 1]
                ) % 12
                chord["root"] = chord["applied"]
                key_scale_intervals = major_scale_intervals

            # Convert scale degrees to pitch offsets
            chord_degree_to_interval = OrderedDict()
            for d in chord_degrees:
                d_abs = (chord["root"] - 1) + (d - 1)
                interval = key_scale_intervals[d_abs % 7]
                interval += 12 * (d_abs // 7)
                chord_degree_to_interval[d] = interval
                # NOTE: Not sure if this is a bug in Hookpad or what?
                if d == 7 and chord["applied"] == 7:
                    chord_degree_to_interval[d] -= 1

            # Apply alterations
            for alt in chord["alterations"]:
                d = int(alt[1:])
                assert d in chord_degree_to_interval
                chord_degree_to_interval[d] += -1 if alt[0] == "b" else 1

            # Create final chord
            result = [key_tonic_pc + v for _, v in chord_degree_to_interval.items()]
            result = Chord(result[0] % 12, tuple(np.diff(result).tolist()))
        return result
