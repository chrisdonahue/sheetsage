import io
import logging
import warnings
from contextlib import redirect_stdout

import jukebox.hparams
import jukebox.make_models
import jukebox.utils.dist_utils
import librosa
import numpy as np
import torch

from ..utils import decode_audio, get_approximate_audio_length
from .base import Representation

_SAMPLE_RATE = 44100
_FRAME_HOP_SIZE = 128
_MIN_LENGTH_SAMPLES = (60 * _SAMPLE_RATE) + 16
_MAX_LENGTH_SAMPLES = (600 * _SAMPLE_RATE) - 96
_CHUNK_FRAMES = 8192
_CHUNK_SAMPLES = _CHUNK_FRAMES * _FRAME_HOP_SIZE

_SINGLETON = None


def init_jukebox_singleton(model="5b", num_layers=53, log=True):
    global _SINGLETON

    if _SINGLETON is None:
        # Set up device
        with redirect_stdout(io.StringIO()) as s:
            rank, local_rank, device = jukebox.utils.dist_utils.setup_dist_from_mpi()
            if log:
                logging.info(s.getvalue())

        # Set up hyperparams
        hps = jukebox.hparams.Hyperparams()
        hps.sr = _SAMPLE_RATE
        hps.n_samples = 3 if model == "5b_lyrics" else 8
        hps.name = "samples"
        chunk_size = 16 if model == "5b_lyrics" else 32
        max_batch_size = 3 if model == "5b_lyrics" else 16
        hps.levels = 3
        hps.hop_fraction = [0.5, 0.5, 0.125]

        # Load VQVAE
        vqvae, *priors = jukebox.make_models.MODELS[model]
        with redirect_stdout(io.StringIO()) as s:
            vqvae = jukebox.make_models.make_vqvae(
                jukebox.hparams.setup_hparams(
                    vqvae, dict(sample_length=_CHUNK_SAMPLES)
                ),
                device,
            )
            if log:
                logging.info(s.getvalue())

        # Set up language model
        if num_layers is not None:
            overrides = dict(prior_depth=num_layers)
        else:
            overrides = dict()
        with redirect_stdout(io.StringIO()) as s:
            lm = jukebox.make_models.make_prior(
                jukebox.hparams.setup_hparams(priors[-1], overrides), vqvae, device
            )
            if log:
                logging.info(s.getvalue())
        lm.prior.only_encode = True

        _SINGLETON = (model, num_layers, hps, vqvae, lm, device)
    else:
        if (model, num_layers) != _SINGLETON[:2]:
            raise Exception("Jukebox can only be initialized once")

    return _SINGLETON


class Jukebox(Representation):
    def __init__(self, num_layers=53, fp16=False, log=True):
        # NOTE: Layer 53 is the deepest that fit on a commodity 12GB card
        (
            _,
            _,
            self.hps,
            self.vqvae,
            self.lm,
            self.device,
        ) = init_jukebox_singleton(model="5b", num_layers=num_layers, log=log)
        self.fp16 = fp16

    @classmethod
    def decode_audio(cls, audio_path, offset=0.0, duration=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(
                audio_path, sr=None, mono=False, offset=offset, duration=duration
            )
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = np.swapaxes(audio, 0, 1)
        audio = np.mean(audio, axis=1, keepdims=False)
        if sr != _SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr,target_sr=_SAMPLE_RATE, res_type="kaiser_best")
        if audio.shape[0] > 0:
            norm_factor = np.abs(audio).max()
            if norm_factor > 0:
                audio /= norm_factor
        return audio

    def _codify_audio(
        self, audio, tqdm=lambda x: x, window_size=_CHUNK_SAMPLES, pad=True
    ):
        # NOTE: Ugly API for legacy test case.
        hop_size = _CHUNK_SAMPLES
        hop_size_frames = window_size // _FRAME_HOP_SIZE
        result = []
        for i in tqdm(list(range(0, audio.shape[0], hop_size))):
            context = audio[i : i + window_size]
            if pad and context.shape[0] < window_size:
                context = np.pad(context, (0, window_size - context.shape[0]))
            with torch.no_grad():
                context = torch.tensor(
                    context, dtype=torch.float32, device=self.device
                ).view(1, -1, 1)
                context_codified = self.vqvae.encode(context)[-1].view(-1).cpu().numpy()
            context_codified = context_codified[:hop_size_frames]
            result.append(context_codified)
        return np.concatenate(result, axis=0)

    def codify_audio(self, audio, tqdm=lambda x: x):
        return self._codify_audio(audio, tqdm=tqdm)

    def lm_activations(
        self,
        audio_codified,
        metadata_offset_seconds=0.0,
        metadata_total_length_seconds=None,
        metadata_artist=None,
        metadata_genre=None,
        metadata_lyrics=None,
        tqdm=lambda x: x,
    ):
        hop_size = _CHUNK_FRAMES
        window_size = _CHUNK_FRAMES
        if audio_codified.shape[0] % _CHUNK_FRAMES != 0:
            raise ValueError()

        # Compute metadata offset
        metadata_initial_offset = int(metadata_offset_seconds * _SAMPLE_RATE)
        metadata_initial_offset = (
            metadata_initial_offset // _FRAME_HOP_SIZE
        ) * _FRAME_HOP_SIZE
        assert metadata_initial_offset % _FRAME_HOP_SIZE == 0
        if metadata_initial_offset < 0:
            raise ValueError()

        # Compute metadata total length
        if metadata_total_length_seconds is None:
            metadata_total_length = audio_codified.shape[0] * _FRAME_HOP_SIZE
        else:
            metadata_total_length = int(metadata_total_length_seconds * _SAMPLE_RATE)
        metadata_total_length = max(metadata_total_length, _MIN_LENGTH_SAMPLES)
        metadata_total_length = min(metadata_total_length, _MAX_LENGTH_SAMPLES)
        metadata_total_length = (
            metadata_total_length // _FRAME_HOP_SIZE
        ) * _FRAME_HOP_SIZE
        assert metadata_total_length % _FRAME_HOP_SIZE == 0
        assert metadata_total_length >= _MIN_LENGTH_SAMPLES
        assert metadata_total_length <= _MAX_LENGTH_SAMPLES

        result = []
        for i in tqdm(list(range(0, audio_codified.shape[0], hop_size))):
            # Select context window
            context = audio_codified[i : i + window_size]
            metadata_offset = metadata_initial_offset + i * _FRAME_HOP_SIZE
            metadata_offset = min(
                metadata_offset,
                metadata_total_length - (context.shape[0] * _FRAME_HOP_SIZE),
            )
            metadata_offset = max(metadata_offset, 0)
            assert metadata_offset % _FRAME_HOP_SIZE == 0

            with torch.no_grad():
                # Context
                x = torch.tensor(context, dtype=torch.int64, device=self.device).view(
                    1, -1
                )

                # Conditioning info
                meta = dict(
                    artist="unknown" if metadata_artist is None else metadata_artist,
                    genre="unknown" if metadata_genre is None else metadata_genre,
                    total_length=metadata_total_length,
                    offset=metadata_offset,
                    lyrics="Placeholder lyrics which do not affect 5b"
                    if metadata_lyrics is None
                    else metadata_lyrics,
                )
                metas = [meta] * self.hps.n_samples
                labels = [None, None, self.lm.labeller.get_batch_labels(metas, "cuda")]
                x_cond, y_cond, _ = self.lm.get_cond(None, self.lm.get_y(labels[-1], 0))
                x_cond = x_cond[:1]
                y_cond = y_cond[:1]

                # Extract activations
                activations = (
                    self.lm.prior.forward(
                        x, x_cond=x_cond, y_cond=y_cond, fp16=self.fp16
                    )
                    .cpu()
                    .numpy()
                )
                if self.fp16:
                    activations = activations.astype(np.float16)
                result.append(activations[0])

                # Clear memory
                del x
                del labels
                del x_cond
                del y_cond
                torch.cuda.empty_cache()

        return np.concatenate(result, axis=0)

    def __call__(self, audio_path, offset=0.0, duration=None):
        audio = self.decode_audio(audio_path, offset=offset, duration=duration)
        if offset == 0.0 and duration is None:
            total_length = audio.shape[0] / _SAMPLE_RATE
        else:
            total_length = get_approximate_audio_length(audio_path)
        codified_audio = self.codify_audio(audio)
        activations = self.lm_activations(
            codified_audio,
            metadata_offset_seconds=offset,
            metadata_total_length_seconds=total_length,
        )
        activations = activations[: int(audio.shape[0] / _FRAME_HOP_SIZE)]
        rate = _SAMPLE_RATE / _FRAME_HOP_SIZE
        return rate, activations
