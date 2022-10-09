import tempfile
from collections import Counter

from ..utils import run_cmd_sync
from .basic import HumanPitchName
from .internal import Harmony, Key, KeyChanges, Melody, MeterChanges


def theorytab_find_applicable(timed_events, search_event, eps=1e-3):
    candidates = [t for t in timed_events if (search_event["beat"] - t["beat"]) > -eps]
    if len(candidates) == 0:
        raise ValueError()
    return candidates[-1]


def estimate_key_changes(meter_changes, harmony, melody):
    meter_changes = MeterChanges(*meter_changes)
    harmony = Harmony(*harmony)
    melody = Melody(*melody)

    # Compute total num tertiary
    meter = meter_changes[0][1]
    assert meter in [(3, 2, 2), (4, 2, 2)]
    tertiary_per_pulse = meter[1] * meter[2]
    tertiary_per_group = meter[0] * tertiary_per_pulse
    total_num_tertiary = 0 if len(harmony) == 0 else harmony[-1][0] + 1
    total_num_tertiary = max(
        total_num_tertiary, 0 if len(melody) == 0 else sum(melody[-1][:2])
    )
    while total_num_tertiary % tertiary_per_group != 0:
        total_num_tertiary += 1

    # Fake tempo
    ppm = 120
    tertiary_to_ms = lambda t: round((t / tertiary_per_pulse) / (ppm / 60) * 1000)

    # "Beat" events
    lines = []
    for t in range(0, total_num_tertiary + tertiary_per_pulse, tertiary_per_pulse):
        strength = 1
        if t % tertiary_per_group == 0:
            strength = 4
        elif meter == (4, 2, 2) and t % (tertiary_per_pulse * 2) == 0:
            strength = 2
        lines.append(("Beat", tertiary_to_ms(t), strength))

    # "Chord" events
    pc_to_melisma_pc = {pc: (2 + (7 * pc)) % 12 for pc in range(12)}
    for i, (t, c) in enumerate(harmony):
        if i == 0:
            t = 0
        if i + 1 < len(harmony):
            d = harmony[i + 1][0] - t
        else:
            d = total_num_tertiary - t
        lines.append(
            ("Chord", tertiary_to_ms(t), tertiary_to_ms(t + d), pc_to_melisma_pc[c[0]])
        )

    # "Note" events
    for t, d, n in melody:
        lines.append(
            ("Note", tertiary_to_ms(t), tertiary_to_ms(t + d), n.as_midi_pitch())
        )

    parameters = """
verbosity=1
default_profile_value = 1.5
npc_or_tpc_profile=0
scoring_mode = 1
segment_beat_level=3
beat_printout_level=2
romnums=0
romnum_type=0
running=0

%CBMS MODEL
major_profile = 5.0 2.0 3.5 2.0 4.5 4.0 2.0 4.5 2.0 3.5 1.5 4.0
minor_profile = 5.0 2.0 3.5 4.5 2.0 4.0 2.0 4.5 3.5 2.0 1.5 4.0
change_penalty=12

%K-S MODEL
%major_profile = 6.35 2.23 3.48 2.33 4.38 4.09 2.52 5.19 2.39 3.66 2.29 2.88
%minor_profile = 6.33 2.68 3.52 5.38 2.60 3.53 2.54 4.75 3.98 2.69 3.34 3.17
%change_penalty = 2.3

%BAYESIAN MODEL
%major_profile = 0.748 0.060 0.488 0.082 0.670 0.460 0.096 0.715 0.104 0.366 0.057 0.400
%minor_profile = 0.712 0.084 0.474 0.618 0.049 0.460 0.105 0.747 0.404 0.067 0.133 0.330
%change_penalty = 0.002
    """.strip()

    formatted = "\n".join(["\t".join([str(a) for a in l]) for l in lines])
    with tempfile.NamedTemporaryFile() as f, tempfile.NamedTemporaryFile() as p:
        with open(f.name, "w") as f:
            f.write(formatted)
        with open(p.name, "w") as p:
            p.write(parameters)
        res, stdout, stderr = run_cmd_sync(
            f"melisma-key -p {p.name} {f.name}", timeout=60
        )
        if res != 0 or len(stderr) > 0:
            raise Exception(f"{stdout}\n{stderr}".strip())

    key_to_count = Counter()
    for key in stdout.split():
        if key.endswith("m"):
            scale = (2, 1, 2, 2, 1, 2)
            key = key[:-1]
        else:
            scale = (2, 2, 1, 2, 2, 2)
        key = Key(HumanPitchName(key).as_pitch_class(), scale)
        key_to_count[key] += 1
    if len(key_to_count) == 0:
        raise Exception("Failed to estimate key")
    key = sorted(key_to_count.keys(), key=lambda k: key_to_count[k])[-1]
    return KeyChanges((0, key))
