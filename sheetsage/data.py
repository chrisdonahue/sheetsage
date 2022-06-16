import gzip
import json
from enum import Enum

from .align import create_beat_to_time_fn
from .assets import retrieve_asset

_EPS = 1e-6


class Split(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class HooktheoryConfig(Enum):
    MELODY_TRANSCRIPTION = 0


class HooktheoryAlignment(Enum):
    USER = 0
    REFINED = 1


class Note:
    def __init__(self, onset, pitch, offset=None):
        if onset < 0:
            raise ValueError()
        if offset is not None and offset <= onset:
            raise ValueError()
        if pitch < 0 or pitch >= 128:
            raise ValueError()
        self.onset = onset
        self.pitch = pitch
        self.offset = offset


class MelodyTranscriptionExample:
    def __init__(self, uid, audio_tag, segment_start, segment_end, melody):
        if segment_start < 0:
            raise ValueError()
        if segment_end <= segment_start:
            raise ValueError()

        melody = sorted(melody, key=lambda n: (n.onset, n.pitch, n.offset))
        if any(
            (n.onset < segment_start - _EPS or n.onset > segment_end + _EPS)
            for n in melody
        ):
            raise ValueError()
        if any(
            n.offset is not None
            and (n.offset < segment_start - _EPS or n.offset > segment_end + _EPS)
            for n in melody
        ):
            raise ValueError(uid)
        for i in range(len(melody) - 1):
            if (
                melody[i].offset is not None
                and melody[i].offset > melody[i + 1].onset + _EPS
            ):
                raise ValueError(uid)

        self.uid = uid
        self.audio_tag = audio_tag
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.melody = melody


_CONFIG_TO_TAGS = {
    HooktheoryConfig.MELODY_TRANSCRIPTION: {
        "require": ["AUDIO_AVAILABLE", "MELODY"],
        # NOTE: Tempo changes are weird on Hooktheory and likely imply a bad alignment
        "deny": ["TEMPO_CHANGES"],
    },
}


def load_hooktheory_raw(
    config=HooktheoryConfig.MELODY_TRANSCRIPTION,
    alignment=HooktheoryAlignment.REFINED,
    additional_required_tags=[],
    additional_denied_tags=[],
):
    if isinstance(config, str):
        config = HooktheoryConfig[config.upper().strip()]
    if isinstance(alignment, str):
        alignment = HooktheoryAlignment[alignment.upper().strip()]

    # Build required tags list
    require = _CONFIG_TO_TAGS[config]["require"]
    require = require + additional_required_tags
    if alignment is not None:
        require.append(
            "USER_ALIGNMENT"
            if alignment == HooktheoryAlignment.USER
            else "REFINED_ALIGNMENT"
        )

    # Build denied tags list
    deny = _CONFIG_TO_TAGS[config]["deny"]
    deny = deny + additional_denied_tags

    # Load dataset
    with gzip.open(retrieve_asset("HOOKTHEORY"), "r") as f:
        hooktheory = json.load(f)

    # Check tags
    all_tags = set()
    for attrs in hooktheory.values():
        for tag in attrs["tags"]:
            all_tags.add(tag)
    for tag in require + deny:
        if tag not in all_tags:
            raise ValueError(f"Invalid tag: {tag}")

    # Filter dataset
    hooktheory = {
        k: v
        for k, v in hooktheory.items()
        if all(tag in v["tags"] for tag in require)
        and all(tag not in v["tags"] for tag in deny)
    }

    return hooktheory


def iter_hooktheory(
    alignment=HooktheoryAlignment.REFINED,
    split=None,
    default_octave=4,
    tqdm=lambda x: x,
    **kwargs,
):
    if isinstance(alignment, str):
        alignment = HooktheoryAlignment[alignment.upper().strip()]
    if isinstance(split, str):
        split = Split[split.upper().strip()]

    hooktheory_raw = load_hooktheory_raw(
        config=HooktheoryConfig.MELODY_TRANSCRIPTION, alignment=alignment
    )
    if split is not None:
        hooktheory_raw = {
            k: v for k, v in hooktheory_raw.items() if v["split"] == split.name
        }

    for uid, attrs in tqdm(hooktheory_raw.items()):
        youtube_id = attrs["youtube"]["id"]
        if youtube_id is None:
            raise Exception("Audio unavailable")

        alignment_ = attrs["alignment"][alignment.name.lower()]
        if alignment_ is None or len(alignment_["times"]) < 2:
            raise Exception("Alignment unavailable")
        beat_to_time = create_beat_to_time_fn(alignment_["beats"], alignment_["times"])
        segment_start = float(beat_to_time(0))
        segment_end = float(beat_to_time(attrs["annotations"]["num_beats"]))

        melody = attrs["annotations"]["melody"]
        if melody is None or len(melody) == 0:
            raise Exception("Melody unavailable")
        melody = [
            Note(
                onset=float(beat_to_time(n["onset"])),
                pitch=(1 + default_octave + n["octave"]) * 12 + n["pitch_class"],
                offset=float(beat_to_time(n["offset"])),
            )
            for n in melody
        ]

        yield MelodyTranscriptionExample(
            uid=uid,
            audio_tag=f"YOUTUBE_{youtube_id}",
            segment_start=segment_start,
            segment_end=segment_end,
            melody=melody,
        )
