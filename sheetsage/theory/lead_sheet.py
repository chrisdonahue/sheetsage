import tempfile
from collections import defaultdict

import numpy as np
import pretty_midi

from ..align import create_beat_to_time_fn
from .internal import (
    Harmony,
    KeyChanges,
    Melody,
    MeterChanges,
    Note,
    TempoChanges,
    _ImmutableIterable,
)
from .theorytab import (
    TheorytabChord,
    TheorytabKey,
    TheorytabMeter,
    TheorytabNote,
    TheorytabTempo,
    TheorytabValueError,
)
from .utils import theorytab_find_applicable

_LILY_TEMPLATE = r"""
#(set-default-paper-size "letter")

<<

\new ChordNames {{
    \set majorSevenSymbol = \markup {{ maj7 }} 
    \set additionalPitchPrefix = #"add"
    \chordmode {{
        {harmony}
    }}
}}

\new Staff {{ 
    {{
        \clef {clef}
        \key {key}
        \time {meter}
        \tempo {tempo}
        {melody}
    }}
}}

>>

\version "2.18.2"
""".strip()


_NUM_SIXTEENTHS_TO_LILY_NAME = {
    1: "16",
    3: "8.",
    2: "8",
    4: "4",
    6: "4.",
    8: "2",
    12: "2.",
    16: "1",
}


class LeadSheet(_ImmutableIterable):
    def __new__(
        cls,
        meter_changes,
        tempo_changes,
        key_changes,
        harmony,
        melody,
        total_num_tertiary=None,
    ):
        # Run value checks
        meter_changes = MeterChanges(*meter_changes)
        tempo_changes = TempoChanges(*tempo_changes)
        key_changes = KeyChanges(*key_changes)
        harmony = Harmony(*harmony)
        melody = Melody(*melody)

        # Compute tertiary per group (measure)
        if len(meter_changes) != 1:
            raise NotImplementedError()
        meter = meter_changes[0][1]
        assert meter in [(3, 2, 2), (4, 2, 2)]

        # TODO: Ensure all changes start on downbeats

        # Compute and/or validate total_num_tertiary
        if total_num_tertiary is None:
            total_num_tertiary = 1
            user_defined = False
        else:
            if not isinstance(total_num_tertiary, int):
                raise TypeError()
            if total_num_tertiary <= 0:
                raise ValueError()
            user_defined = True
        for l in [meter_changes, tempo_changes, key_changes, harmony, melody]:
            if len(l) == 0:
                continue
            if len(l[-1]) == 2:
                o, _ = l[-1]
                d = 1
            else:
                o, d, _ = l[-1]
            if (o + d) > total_num_tertiary:
                if user_defined:
                    raise ValueError()
                else:
                    total_num_tertiary = o + d

        # Round up to nearest measure
        # NOTE: We always need at least one measure because key/meter are defined at the beginning
        tertiary_per_group = int(np.prod(meter))
        while total_num_tertiary % tertiary_per_group != 0:
            total_num_tertiary += 1
        assert isinstance(total_num_tertiary, int)
        assert total_num_tertiary % tertiary_per_group == 0
        assert total_num_tertiary // tertiary_per_group > 0

        return super().__new__(
            cls,
            (
                meter_changes,
                tempo_changes,
                key_changes,
                harmony,
                melody,
                total_num_tertiary,
            ),
        )

    def as_lily(
        self, clef="treble", adjust_melody_octave=True, skip_unknown_chords=False
    ):
        if clef not in ["treble", "bass"]:
            raise ValueError()
        (
            meter_changes,
            tempo_changes,
            key_changes,
            harmony,
            melody,
            total_num_tertiary,
        ) = self

        # Format meter
        assert len(meter_changes) == 1
        meter = meter_changes[0][1]
        assert meter in [(3, 2, 2), (4, 2, 2)]
        tertiary_per_group = int(np.prod(meter))
        meter_lily = meter.as_lily()
        meter_lily = f"{meter_lily[0]}/{meter_lily[1]}"

        # Format tempo
        if len(tempo_changes) != 1:
            raise NotImplementedError()
        assert len(tempo_changes) == 1
        tempo = tempo_changes[0][1]
        tempo_lily = tempo.as_lily(meter)
        tempo_lily = f"{tempo_lily[0]} = {tempo_lily[1]}"

        # Format key
        if len(key_changes) != 1:
            raise NotImplementedError()
        key = key_changes[0][1]
        key_lily = key.as_lily()
        key_lily = f"{key_lily[0]} \\{key_lily[1]}"

        # Add in rests in between chords
        chords_and_rests = []
        if len(harmony) == 0:
            chords_and_rests.append((total_num_tertiary, None))
        else:
            for i, (t, c) in enumerate(harmony):
                if i == 0 and t > 0:
                    chords_and_rests.append((t, None))
                if i + 1 < len(harmony):
                    d = harmony[i + 1][0] - t
                else:
                    d = total_num_tertiary - t
                chords_and_rests.append((d, c))
        assert all(d > 0 for d, _ in chords_and_rests)
        assert sum(d for d, _ in chords_and_rests) == total_num_tertiary

        # Format chords
        harmony = []
        for d, c in chords_and_rests:
            # NOTE: c is None means no chord *change*, cs is (None, None) means *change* to N.C.
            if c is None:
                lpc = ("s", "")
            else:
                lpc = c.as_lily(key)
            if not isinstance(lpc[1], str):
                assert isinstance(lpc[1], tuple)
                if not skip_unknown_chords:
                    raise ValueError("Unknown chord")
                lpc = ("s", "")
            lp = f"{lpc[0]}16*{d}"
            if len(lpc[1]) > 0:
                lp += ":" + lpc[1]
            harmony.append(lp)
        harmony_lily = " ".join(harmony)

        # Adjust melody to be centered in clef
        # NOTE: Finds the octave where the most melody are on the staff lines
        if adjust_melody_octave and len(melody) > 0:
            if clef == "treble":
                midi_pitch_range = (63, 78)  # Eb at bottom of treble clef, F# at top
            elif clef == "bass":
                midi_pitch_range = (42, 58)  # Gb at bottom of bass clef, A# at top
            else:
                assert False
            midi_pitches = np.array([ns.as_midi_pitch() for _, _, ns in melody])
            candidate_octaves = np.arange(-1000, 1000)
            midi_pitches_adjusted = (candidate_octaves * 12)[
                :, np.newaxis
            ] + midi_pitches[np.newaxis, :]
            midi_pitches_onstaff = np.logical_and(
                midi_pitches_adjusted >= midi_pitch_range[0],
                midi_pitches_adjusted <= midi_pitch_range[1],
            )
            best_octave = int(
                candidate_octaves[
                    np.argmax(midi_pitches_onstaff.astype(np.int64).sum(axis=1))
                ]
            )
            melody = [(s, d, Note(ns[0], ns[1] + best_octave)) for s, d, ns in melody]

        # Add in rests in between melody
        last_offset = 0
        notes_and_rests = []
        for t, d, ns in melody:
            assert t >= last_offset
            if t != last_offset:
                notes_and_rests.append((t - last_offset, None))
            notes_and_rests.append((d, ns))
            last_offset = t + d
        t = total_num_tertiary
        assert t >= last_offset
        if t != last_offset:
            notes_and_rests.append((t - last_offset, None))
        assert all(d > 0 for d, _ in notes_and_rests)
        assert sum(d for d, _ in notes_and_rests) == total_num_tertiary

        # Beaming logic
        bar_to_notes = defaultdict(list)
        t = 0
        for d, ns in notes_and_rests:
            tied = False
            while d > 0:
                bar = t // tertiary_per_group
                bar_remaining = ((bar + 1) * tertiary_per_group) - t
                consumed = min(d, bar_remaining)
                if consumed not in _NUM_SIXTEENTHS_TO_LILY_NAME:
                    while _NUM_SIXTEENTHS_TO_LILY_NAME.get(consumed, ".").endswith("."):
                        consumed -= 1
                d -= consumed
                t += consumed
                if ns is None:
                    lp = "r"
                else:
                    lp = "".join(ns.as_lily(key))
                lp += _NUM_SIXTEENTHS_TO_LILY_NAME[consumed]
                lp += "~" if d > 0 else ""
                bar_to_notes[bar].append(lp)
                tied = True

        # Format notes
        melody_lily = " | ".join([" ".join(notes) for _, notes in bar_to_notes.items()])

        return _LILY_TEMPLATE.format(
            clef=clef,
            key=key_lily,
            meter=meter_lily,
            tempo=tempo_lily,
            harmony=harmony_lily,
            melody=melody_lily,
        )

    def as_midi(self, pulse_to_time_fn=None, adjust_melody_octave=True):
        (
            meter_changes,
            tempo_changes,
            _,
            harmony,
            melody,
            total_num_tertiary,
        ) = self

        # Assumptions
        assert len(meter_changes) == 1
        meter = meter_changes[0][1]
        assert meter in [(3, 2, 2), (4, 2, 2)]
        tertiary_per_group = int(np.prod(meter))
        tertiary_per_pulse = int(np.prod(meter[1:]))

        # Create tertiary_to_time_fn
        if pulse_to_time_fn is None:
            if len(tempo_changes) != 1:
                raise NotImplementedError()
            tempo = tempo_changes[0][1]
            pps = tempo[0] / 60
            tertiaries = [0, tertiary_per_pulse]
            times = [0, 1 / pps]
            tertiary_to_time_fn = create_beat_to_time_fn(tertiaries, times)
        else:
            tertiary_to_time_fn = lambda t: pulse_to_time_fn(t / tertiary_per_pulse)

        # Adjust melody to be mostly in midi octave 5
        if adjust_melody_octave and len(melody) > 0:
            midi_pitch_range = (60, 71)
            midi_pitches = np.array([ns.as_midi_pitch() for _, _, ns in melody])
            candidate_octaves = np.arange(-1000, 1000)
            midi_pitches_adjusted = (candidate_octaves * 12)[
                :, np.newaxis
            ] + midi_pitches[np.newaxis, :]
            midi_pitches_onstaff = np.logical_and(
                midi_pitches_adjusted >= midi_pitch_range[0],
                midi_pitches_adjusted <= midi_pitch_range[1],
            )
            best_octave = int(
                candidate_octaves[
                    np.argmax(midi_pitches_onstaff.astype(np.int64).sum(axis=1))
                ]
            )
            melody = [(s, d, Note(ns[0], ns[1] + best_octave)) for s, d, ns in melody]

        # Create click
        click = pretty_midi.Instrument(program=0, is_drum=True)
        for t in range(0, total_num_tertiary, tertiary_per_pulse):
            velocity = 75
            pitch = 31

            # Downbeat
            if t % tertiary_per_group == 0:
                velocity = 100
                pitch = 37

            click.notes.append(
                pretty_midi.Note(
                    velocity,
                    pitch,
                    tertiary_to_time_fn(t),
                    tertiary_to_time_fn(t + tertiary_per_pulse),
                )
            )

        # Create harmony
        harmony_ins = pretty_midi.Instrument(program=24)  # Acoustic Guitar (nylon)
        for i, (t, c) in enumerate(harmony):
            if i + 1 < len(harmony):
                d = harmony[i + 1][0] - t
            else:
                d = total_num_tertiary - t
            for p in c.as_midi_pitches():
                harmony_ins.notes.append(
                    pretty_midi.Note(
                        67,
                        p,
                        tertiary_to_time_fn(t),
                        tertiary_to_time_fn(t + d),
                    )
                )

        # Create melody
        melody_ins = pretty_midi.Instrument(program=0)
        for t, d, ns in melody:
            melody_ins.notes.append(
                pretty_midi.Note(
                    100,
                    ns.as_midi_pitch(),
                    tertiary_to_time_fn(t),
                    tertiary_to_time_fn(t + d),
                )
            )

        # Create MIDI
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.extend([click, harmony_ins, melody_ins])
        with tempfile.NamedTemporaryFile() as f:
            midi.write(f.name)
            with open(f.name, "rb") as f:
                return f.read()

    @classmethod
    def from_theorytab(
        cls,
        analysis,
        ignore_inversion=True,
        skip_bad_notes_and_chords=False,
    ):
        # For theorytab we always subdivide into 1/4 beat
        beat_to_tertiary = lambda b: round(b * 4)
        end_beat = analysis["endBeat"] - 1

        # Meters
        meter_changes = []
        last_meter = None
        for ttm in analysis["meters"]:
            ttm = TheorytabMeter(ttm)
            meter = ttm.as_meter()
            if meter != last_meter:
                meter_changes.append((beat_to_tertiary(ttm["beat"] - 1), meter))
            last_meter = meter

        # Tempos
        tempo_changes = []
        last_tempo = None
        for ttt in analysis["tempos"]:
            ttt = TheorytabTempo(ttt)
            tempo = ttt.as_tempo()
            if tempo != last_tempo:
                tempo_changes.append((beat_to_tertiary(ttt["beat"] - 1), tempo))
            last_tempo = tempo

        # Keys
        ttk_keys = []
        key_changes = []
        last_key = None
        for ttk in analysis["keys"]:
            ttk = TheorytabKey(ttk)
            ttk_keys.append(ttk)
            key = ttk.as_key()
            if key != last_key:
                key_changes.append((beat_to_tertiary(ttk["beat"] - 1), key))
            last_key = key

        # Chords
        harmony = []
        last_chord = None
        for ttc in analysis["chords"]:
            try:
                ttc = TheorytabChord(ttc)
            except TheorytabValueError as e:
                if not skip_bad_notes_and_chords:
                    raise e
                continue
            if ttc.will_sound():
                chord = ttc.as_chord(
                    theorytab_find_applicable(ttk_keys, ttc),
                    root_position=ignore_inversion,
                )
                if chord != last_chord:
                    ob = ttc["beat"] - 1
                    db = ttc["duration"]
                    if ob + db > (end_beat + 1e-6):
                        raise ValueError()
                    harmony.append((beat_to_tertiary(ob), chord))
                last_chord = chord

        # Notes
        melody = []
        for ttn in analysis["notes"]:
            try:
                ttn = TheorytabNote(ttn)
            except TheorytabValueError as e:
                if not skip_bad_notes_and_chords:
                    raise e
                continue
            if ttn.will_sound():
                ob = ttn["beat"] - 1
                db = ttn["duration"]
                if ob + db > (end_beat + 1e-6):
                    raise ValueError()
                melody.append(
                    (
                        beat_to_tertiary(ob),
                        beat_to_tertiary(db),
                        ttn.as_note(theorytab_find_applicable(ttk_keys, ttn)),
                    )
                )

        # Trim extra changes
        total_num_tertiary = beat_to_tertiary(end_beat)
        meter_changes = [m for m in meter_changes if m[0] < total_num_tertiary]
        tempo_changes = [t for t in tempo_changes if t[0] < total_num_tertiary]
        key_changes = [k for k in key_changes if k[0] < total_num_tertiary]

        return cls(
            meter_changes,
            tempo_changes,
            key_changes,
            harmony,
            melody,
            total_num_tertiary=total_num_tertiary,
        )
