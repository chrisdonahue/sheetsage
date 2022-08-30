import itertools
import json
import logging
import tempfile
from enum import Enum
from functools import lru_cache as cache

import numpy as np
import torch

from .align import create_beat_to_time_fn
from .assets import retrieve_asset
from .beat_track import madmom
from .modules import EncOnlyTransducer, IdentityEncoder, TransformerEncoder
from .representations import Handcrafted, Jukebox
from .theory import (
    Chord,
    Harmony,
    KeyChanges,
    LeadSheet,
    Melody,
    MeterChanges,
    Note,
    TempoChanges,
    estimate_key_changes,
)
from .utils import decode_audio, get_approximate_audio_length, retrieve_audio_bytes


class InputFeats(Enum):
    HANDCRAFTED = 0
    JUKEBOX = 1


class Task(Enum):
    MELODY = 0
    HARMONY = 1


class Model(Enum):
    LINEAR = 0
    TRANSFORMER = 1


class OutputModality(Enum):
    MELODY_MIDI = 0
    LEAD_SHEET = 1


_INPUT_TO_FRAME_RATE = {
    InputFeats.HANDCRAFTED: 16000 / 512,
    InputFeats.JUKEBOX: 44100 / 128,
}
_INPUT_TO_DIM = {
    InputFeats.HANDCRAFTED: 229,
    InputFeats.JUKEBOX: 4800,
}
_JUKEBOX_CHUNK_DURATION_EDGE = 23.75
_TERTIARIES_PER_BEAT = 4
_MELODY_PITCH_MIN = 21
_HARMONY_FAMILIES = ["", "m", "m7", "7", "maj7", "sus", "dim", "aug"]
_FAMILY_TO_INTERVALS = {
    "": (4, 3),
    "m": (3, 4),
    "m7": (3, 4, 3),
    "7": (4, 3, 3),
    "maj7": (4, 3, 4),
    "sus": (5, 2),
    "dim": (3, 3),
    "aug": (4, 4),
}
_TASK_TO_VOCAB_SIZE = {Task.MELODY: 89, Task.HARMONY: 97}
_MAX_TERTIARIES_PER_CHUNK = 384


@cache()
def _init_extractor(input_feats):
    if input_feats == InputFeats.HANDCRAFTED:
        extractor = Handcrafted()
    elif input_feats == InputFeats.JUKEBOX:
        extractor = Jukebox()
    else:
        raise ValueError()
    return extractor


@cache()
def _init_model(task, input_feats, model):
    if model == Model.LINEAR:
        # NOTE: Just need to catalogue these configs / weights
        raise NotImplementedError()

    asset_prefix = f"SHEETSAGE_V02_{input_feats.name}_{task.name}"
    with open(retrieve_asset(f"{asset_prefix}_CFG"), "r") as f:
        cfg = json.load(f)
    assert cfg["src_max_len"] == _MAX_TERTIARIES_PER_CHUNK

    src_dim = _INPUT_TO_DIM[input_feats]
    output_dim = _TASK_TO_VOCAB_SIZE[task]

    if cfg["model"] == "probe":
        model = EncOnlyTransducer(
            output_dim,
            src_emb_mode="identity",
            src_vocab_size=None,
            src_dim=src_dim,
            src_emb_dim=None,
            src_pos_emb=False,
            src_dropout_p=0.0,
            enc_cls=IdentityEncoder,
            enc_kwargs={},
        )
    elif cfg["model"] == "transformer":
        model = EncOnlyTransducer(
            output_dim,
            src_emb_mode="project",
            src_vocab_size=None,
            src_dim=src_dim,
            src_emb_dim=512,
            src_pos_emb="pos_emb" in cfg["hacks"],
            src_dropout_p=0.1,
            enc_cls=TransformerEncoder,
            enc_kwargs={
                "model_dim": 512,
                "num_heads": 8,
                "num_layers": 4 if "4layers" in cfg["hacks"] else 6,
                "feedforward_dim": 2048,
                "dropout_p": 0.1,
            },
        )
    else:
        raise ValueError()

    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(
        torch.load(retrieve_asset(f"{asset_prefix}_MODEL"), map_location=device)
    )
    model.eval()
    return model


def _closest_idx(x, l):
    assert len(l) > 0
    return np.argmin([abs(li - x) for li in l])


def sheetsage(
    audio_path_or_bytes_url,
    segment_start_hint=None,
    segment_end_hint=None,
    segment_hints_are_downbeats=False,
    measures_per_segment=8,
    input_feats=InputFeats.JUKEBOX,
    output_modality=OutputModality.LEAD_SHEET,
    beat_detection_padding=15.0,
    beats_per_measure_hint=None,
):
    """Main driver function for Sheet Sage.

    Parameters
    ----------
    audio_path_or_bytes_url : :class:`pathlib.Path`, bytes, or str
       The filepath, raw bytes, or string URL of the audio file.
    segment_start_hint : float or None
       Pass
    segment_end_hint : float or None
       Pass
    input_feats : :class:`Input`
    output_modality : :class:`Output`
    measures_per_segment : int
    beat_detection_padding : float
    beats_per_measure_hint : int or None

    Raises
    ------

    Returns
    -------
    """
    # Check arguments
    if segment_start_hint is not None:
        if segment_start_hint < 0:
            raise ValueError("Segment start hint must be non-negative")
    if segment_end_hint is not None:
        if segment_end_hint < 0:
            raise ValueError("Segment end hint must be non-negative")
        if segment_start_hint is None:
            raise ValueError("Must specify a start hint with end hint")
    if segment_start_hint is not None and segment_end_hint is not None:
        if segment_end_hint <= segment_start_hint:
            raise ValueError("Segment end before segment start")
    if measures_per_segment <= 0:
        raise ValueError("Invalid measures per segment specified")
    if measures_per_segment > 24:
        raise ValueError("Sheet Sage can only transcribe 24 measures per segment")
    if beats_per_measure_hint is not None and beats_per_measure_hint not in [3, 4]:
        raise ValueError(
            "Currently, Sheet Sage only supports 4/4 and 3/4 time signatures"
        )

    # Download audio if URL specified
    audio_path_or_bytes = audio_path_or_bytes_url
    if isinstance(audio_path_or_bytes_url, str):
        logging.info(f"Retrieving audio from {audio_path_or_bytes_url}")
        audio_path_or_bytes = retrieve_audio_bytes(audio_path_or_bytes_url)

    logging.info("Detecting beats")

    # Run beat detection
    beat_detection_start = 0.0 if segment_start_hint is None else segment_start_hint
    beat_detection_start = max(beat_detection_start - beat_detection_padding, 0.0)
    beat_detection_end = None if segment_end_hint is None else segment_end_hint
    beat_detection_end = (
        None
        if beat_detection_end is None
        else beat_detection_end + beat_detection_padding
    )
    sr, audio = decode_audio(
        audio_path_or_bytes,
        offset=beat_detection_start,
        duration=None
        if beat_detection_end is None
        else beat_detection_end - beat_detection_start,
    )
    first_downbeat_idx, beats_per_measure, beats = madmom(
        sr,
        audio,
        beats_per_bar=beats_per_measure_hint
        if beats_per_measure_hint is not None
        else [3, 4],
    )
    if first_downbeat_idx is None or beats_per_measure is None or len(beats) == 0:
        raise ValueError("Audio too short to detect time signature")
    assert first_downbeat_idx >= 0 and first_downbeat_idx < beats_per_measure
    assert beats_per_measure in [3, 4]
    beats = [beat_detection_start + t for t in beats]
    downbeats = [
        t
        for i, t in enumerate(beats)
        if i % beats_per_measure == first_downbeat_idx % beats_per_measure
    ]
    assert len(beats) > 0
    assert len(downbeats) > 0

    # Convert beats into tertiary (sixteenth note) timestamps
    # NOTE: Yes, this is super ugly, but sometimes you gotta do what works :shrug:
    beat_to_time_fn = create_beat_to_time_fn(list(range(len(beats))), beats)
    tertiaries_raw = np.arange(0, len(beats) - 1 + 1e-6, 1 / _TERTIARIES_PER_BEAT)
    assert tertiaries_raw.shape[0] == (len(beats) - 1) * _TERTIARIES_PER_BEAT + 1
    tertiaries = tertiaries_raw - (1 / _TERTIARIES_PER_BEAT) / 2
    tertiary_times = beat_to_time_fn(tertiaries)
    tertiary_times = np.maximum(tertiary_times, 0.0)
    tertiary_times = np.minimum(tertiary_times, beats[-1])
    tertiary_diff_frames = np.diff(tertiary_times) * _INPUT_TO_FRAME_RATE[input_feats]
    if np.any(tertiary_diff_frames.astype(np.int64) == 0):
        raise ValueError("Tempo too fast for beat-informed feature resampling")

    # Find first downbeat of the song from optional hint
    if segment_start_hint is not None:
        if segment_hints_are_downbeats:
            first_downbeat = segment_start_hint
        else:
            first_downbeat = downbeats[_closest_idx(segment_start_hint, downbeats)]
        first_downbeat_idx = _closest_idx(first_downbeat, beats)
        downbeats = [
            t
            for i, t in enumerate(beats)
            if i % beats_per_measure == first_downbeat_idx % beats_per_measure
        ]

    # Find last downbeat of the song from optional hint
    if segment_end_hint is None:
        last_downbeat = downbeats[-1]
    else:
        if segment_hints_are_downbeats:
            last_downbeat = segment_end_hint
        else:
            last_downbeat = downbeats[_closest_idx(segment_end_hint, downbeats)]
    last_downbeat_idx = _closest_idx(last_downbeat, beats)

    # Identify suitable chunks for running through transcription model
    tertiary_chunks = []
    beats_per_segment = beats_per_measure * measures_per_segment
    tertiaries_per_segment = _TERTIARIES_PER_BEAT * beats_per_segment
    for beat_idx in range(first_downbeat_idx, last_downbeat_idx, beats_per_segment):
        tertiary_start_idx = beat_idx * _TERTIARIES_PER_BEAT
        tertiary_end_idx = ((beat_idx + beats_per_segment) * _TERTIARIES_PER_BEAT) + 1
        tertiary_end_idx = min(tertiary_end_idx, tertiary_times.shape[0])
        duration = (
            tertiary_times[tertiary_end_idx - 1] - tertiary_times[tertiary_start_idx]
        )
        assert duration > 0
        if duration > _JUKEBOX_CHUNK_DURATION_EDGE:
            raise NotImplementedError(
                "Dynamic chunking not implemented. Try halving measures_per_segment."
            )
        tertiary_chunks.append((tertiary_start_idx, tertiary_end_idx))

    # Extract features
    logging.info("Extracting feats")
    extractor = _init_extractor(input_feats)
    features = []
    with tempfile.NamedTemporaryFile("wb") as f:
        if isinstance(audio_path_or_bytes, bytes):
            f.write(audio_path_or_bytes)
            f.flush()
            audio_path = f.name
        else:
            audio_path = audio_path_or_bytes

        for tertiary_start_idx, tertiary_end_idx in tertiary_chunks:
            chunk_tertiaries = tertiary_times[tertiary_start_idx:tertiary_end_idx]
            offset = chunk_tertiaries[0]
            duration = chunk_tertiaries[-1] - offset
            assert duration <= _JUKEBOX_CHUNK_DURATION_EDGE
            fr, feats = extractor(audio_path, offset=offset, duration=duration)

            beat_resampled = []
            for i in range(chunk_tertiaries.shape[0] - 1):
                s = int((chunk_tertiaries[i] - offset) * fr)
                e = int((chunk_tertiaries[i + 1] - offset) * fr)
                assert e > s
                beat_resampled.append(np.mean(feats[s:e], axis=0, keepdims=True))
            beat_resampled = np.concatenate(beat_resampled, axis=0)

            features.append(beat_resampled)

    global _PROBE
    _PROBE = np.copy(features[0])

    # Normalize handcrafted features (after beat resampling)
    # NOTE: Normalizing after beat resampling is probably a bug in retrospect, but it's
    # what the model expects.
    if input_feats == InputFeats.HANDCRAFTED:
        moments = np.load(retrieve_asset("SHEETSAGE_V02_HANDCRAFTED_MOMENTS"))
        for chunk in features:
            chunk -= moments[0]
            chunk /= moments[1]

    # Transcribe chunks
    logging.info("Transcribing")
    melody_model = _init_model(Task.MELODY, input_feats, Model.TRANSFORMER)
    melody_logits = []
    if output_modality == OutputModality.LEAD_SHEET:
        harmony_model = _init_model(Task.HARMONY, input_feats, Model.TRANSFORMER)
        harmony_logits = []
    device = torch.device("cpu")
    with torch.no_grad():
        for src in features:
            src_len = src.shape[0]
            src = np.pad(src, [(0, _MAX_TERTIARIES_PER_CHUNK - src_len), (0, 0)])
            src = src[:, np.newaxis]
            src = torch.tensor(src).float()
            src_len = torch.tensor(src_len).long().view(-1)
            src.to(device)
            src_len.to(device)

            chunk_melody_logits = melody_model(src, src_len, None, None)[
                : src_len.item(), 0
            ]
            melody_logits.append(chunk_melody_logits.cpu().numpy())
            if output_modality == OutputModality.LEAD_SHEET:
                chunk_harmony_logits = harmony_model(src, src_len, None, None)[
                    : src_len.item(), 0
                ]
                harmony_logits.append(chunk_harmony_logits.cpu().numpy())
    melody_logits = np.concatenate(melody_logits, axis=0)
    if output_modality == OutputModality.LEAD_SHEET:
        harmony_logits = np.concatenate(harmony_logits, axis=0)
    total_num_tertiary = melody_logits.shape[0]

    logging.info("Formatting output")

    # Decode melody
    melody_preds = np.argmax(melody_logits, axis=-1)
    melody_onsets = []
    for o, p in enumerate(melody_preds):
        if p != 0:
            assert p >= 1
            p -= 1
            p = (p + _MELODY_PITCH_MIN).tolist()
            melody_onsets.append((o, Note(p % 12, p // 12)))
    melody = []
    for i, (o, n) in enumerate(melody_onsets):
        if i + 1 < len(melody_onsets):
            d = melody_onsets[i + 1][0] - o
        else:
            d = total_num_tertiary - o
        melody.append((o, d, n))
    melody = Melody(*melody)

    # Decode harmony
    if output_modality == OutputModality.LEAD_SHEET:
        harmony_preds = np.argmax(harmony_logits, axis=-1)
        harmony = []
        last_chord = None
        for o, c in enumerate(harmony_preds):
            if c != 0:
                assert c >= 1
                c -= 1
                c = c.tolist()
                c = (
                    c // len(_HARMONY_FAMILIES),
                    _HARMONY_FAMILIES[c % len(_HARMONY_FAMILIES)],
                )
                chord = Chord(c[0], _FAMILY_TO_INTERVALS[c[1]])
                if chord != last_chord:
                    harmony.append((o, chord))
                last_chord = chord
        harmony = Harmony(*harmony)

    # Create lead sheet
    meter_changes = MeterChanges((0, (beats_per_measure, 2, 2)))
    tempo_changes = TempoChanges((0, (120,)))
    if len(melody) == 0:
        # NOTE: C major by default
        key_changes = KeyChanges((0, (0, (2, 2, 1, 2, 2, 2))))
    else:
        key_changes = estimate_key_changes(meter_changes, harmony, melody)
    lead_sheet = LeadSheet(
        meter_changes, tempo_changes, key_changes, harmony, melody, total_num_tertiary
    )

    return lead_sheet
