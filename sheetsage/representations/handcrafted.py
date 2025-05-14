import pickle

import librosa
import numpy as np

from ..utils import decode_audio
from .base import Representation


class OAFMelSpec(Representation):
    # NOTE: This configuration is from Onsets & Frames (Hawthorne et al. 17).
    # https://github.com/magenta/magenta/blob/9885adef56d134763a89de5584f7aa18ca7d53b6/magenta/models/onsets_frames_transcription/constants.py
    # https://github.com/magenta/magenta/blob/9885adef56d134763a89de5584f7aa18ca7d53b6/magenta/models/onsets_frames_transcription/data.py#L89
    _SR = 16000
    _NFFT = 2048
    _HOP_SIZE = 512
    _FMIN = 30.0
    _NMELS = 229
    _HTK = False
    _LOG = True

    def __call__(self, audio_path, offset=0.0, duration=None):
        sr, audio = decode_audio(
            audio_path,
            sr=self._SR,
            offset=offset,
            duration=duration,
            mono=True,
            normalize=False,
        )
        features = librosa.feature.melspectrogram(
            y=audio[:, 0],
            sr=self._SR,
            n_fft=self._NFFT,
            hop_length=self._HOP_SIZE,
            fmin=self._FMIN,
            n_mels=self._NMELS,
            htk=self._HTK,
        ).T
        features = features.astype(np.float32)
        if self._LOG:
            features = librosa.power_to_db(features)
        return self._SR / self._HOP_SIZE, features
