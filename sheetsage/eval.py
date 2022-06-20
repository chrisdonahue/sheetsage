import copy
import logging
import pathlib
import shutil
import tempfile
import warnings
from io import BytesIO

import mir_eval
import numpy as np
import pretty_midi

from .data import MelodyTranscriptionExample, as_pretty_midi

# NOTE: This is the standard alignment tolerance used in most transcription literature
EVAL_TOLERANCE = 0.050


def _trim_midi(midi, segment_start, segment_end, tolerance=0):
    if tolerance is not None and tolerance > 0:
        segment_start -= tolerance
        segment_end += tolerance
    num_dropped = 0
    for i in midi.instruments:
        num_notes = len(i.notes)
        i.notes = [
            n for n in i.notes if n.start >= segment_start and n.start <= segment_end
        ]
        num_dropped += num_notes - len(i.notes)
    return midi, num_dropped


def _midi_to_mir_eval(midi, dummy_offsets=True):
    notes = []
    for i in midi.instruments:
        if i.is_drum:
            continue
        for n in i.notes:
            notes.append((n.start, n.end, n.pitch))
    notes = sorted(notes)
    note_onsets = [s for s, _, _ in notes]
    note_offsets = [e for _, e, _ in notes]
    if dummy_offsets and len(note_onsets) > 0:
        note_offsets = note_onsets[1:] + [note_onsets[-1] + 1]
    intervals = np.stack([note_onsets, note_offsets], axis=1).astype(np.float64)
    pitches = np.array([p for _, _, p in notes], dtype=np.int64)
    return intervals, pitches


def _mir_eval_onset_prf(
    ref_intervals, ref_pitches, est_intervals, est_pitches, tolerance=EVAL_TOLERANCE
):
    m_to_f = lambda m: 440.0 * np.power(2, (m.astype(np.float32) - 69) / 12)
    with warnings.catch_warnings():
        # NOTE: This function warns / returns zero when ref is empty
        warnings.simplefilter("ignore")
        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            m_to_f(ref_pitches),
            est_intervals,
            m_to_f(est_pitches),
            onset_tolerance=tolerance,
            pitch_tolerance=1.0,
            offset_ratio=None,
        )
    return p, r, f1


def f1(
    ref_midi,
    est_midi,
    tolerance=EVAL_TOLERANCE,
    octave_invariant_radius=16,
):
    ref_midi = as_pretty_midi(ref_midi)
    est_midi = as_pretty_midi(est_midi)

    # Copy for safety
    ref_midi = copy.deepcopy(ref_midi)
    est_midi = copy.deepcopy(est_midi)

    # Sanity check reference MIDI
    ref_example = MelodyTranscriptionExample.from_midi(ref_midi)

    # Remove drums
    ref_midi.instruments = [i for i in ref_midi.instruments if not i.is_drum]
    est_midi.instruments = [i for i in est_midi.instruments if not i.is_drum]
    if len(est_midi.instruments) > 1:
        warnings.warn(f"Multiple ({len(est_midi.instruments)}) instruments detected")

    # Trim MIDI
    est_midi, num_dropped = _trim_midi(
        est_midi,
        ref_example.segment_start,
        ref_example.segment_end,
        tolerance=tolerance,
    )
    if num_dropped > 0:
        warnings.warn(f"{num_dropped} notes outside of segment")

    # Convert to mir_eval-style
    ref_intervals, ref_pitches = _midi_to_mir_eval(ref_midi, dummy_offsets=False)
    est_intervals, est_pitches = _midi_to_mir_eval(est_midi, dummy_offsets=False)

    # Octave-invariant evaluation
    octaves = list(range(-octave_invariant_radius, octave_invariant_radius + 1))
    ps = []
    rs = []
    f1s = []
    for o in octaves:
        p, r, f1 = _mir_eval_onset_prf(
            ref_intervals,
            (o * 12) + ref_pitches,
            est_intervals,
            est_pitches,
            tolerance=tolerance,
        )
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    best_octave_idx = np.argmax(f1s)
    return (
        f1s[best_octave_idx],
        ps[best_octave_idx],
        rs[best_octave_idx],
        octaves[best_octave_idx],
    )


def eval_dataset(ref, est, allow_abstain=False, return_detail=False):
    ref = pathlib.Path(ref)
    est = pathlib.Path(est)
    detail = {}
    num_abstain = 0
    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as est_dir:
        if ref.is_file():
            shutil.unpack_archive(str(ref), ref_dir)
            ref = pathlib.Path(ref_dir)
        if est.is_file():
            shutil.unpack_archive(str(est), est_dir)
            est = pathlib.Path(est_dir)
        if not ref.is_dir():
            raise Exception("Reference directory not found")
        if not est.is_dir():
            raise Exception("Estimated directory not found")

        ref_uid_to_path = {p.stem: p for p in sorted(ref.glob("*.mid*"))}
        est_uid_to_path = {p.stem: p for p in sorted(est.glob("*.mid*"))}
        for uid, ref_path in ref_uid_to_path.items():
            est_path = est_uid_to_path.get(uid)
            if est_path is None:
                if allow_abstain:
                    num_abstain += 1
                    detail[uid] = "ABSTAINED"
                    continue
                else:
                    raise Exception("Abstaining not allowed")

            f1_, p, r, octave_shift = f1(
                pretty_midi.PrettyMIDI(str(ref_path)),
                pretty_midi.PrettyMIDI(str(est_path)),
            )
            detail[uid] = {"f1": f1_, "p": p, "r": r, "octave_shift": octave_shift}

    if num_abstain > 0:
        assert allow_abstain
        warnings.warn(f"Abstained on {num_abstain} examples")

    f1_ = np.mean([d["f1"] for d in detail.values() if isinstance(d, dict)])
    result = f1_
    if return_detail:
        result = (f1_, detail)
    return result


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("ref_directory_or_archive", type=str)
    parser.add_argument("est_directory_or_archive", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--allow_abstain", action="store_true")

    parser.set_defaults(output_path=None, allow_abstain=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result, detailed = eval_dataset(
        args.ref_directory_or_archive,
        args.est_directory_or_archive,
        return_detail=True,
        allow_abstain=args.allow_abstain,
    )
    logging.info(f"Overall score: {result}")

    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            f.write(json.dumps(detailed, indent=2))
