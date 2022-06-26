import gzip
import hashlib
import pathlib
import shlex
import subprocess
import tempfile
import warnings

import audioread
import librosa
import numpy as np
from scipy.io.wavfile import write as wavwrite


def compute_checksum(path_or_bytes, algorithm="sha256", gunzip=False, chunk_size=4096):
    """Computes checksum of target path.

    Parameters
    ----------
    path_or_bytes : :class:`pathlib.Path` or bytes
       Location or bytes of file to compute checksum for.
    algorithm : str, optional
       Hash algorithm (from :func:`hashlib.algorithms_available`); default ``sha256``.
    gunzip : bool, optional
       If true, decompress before computing checksum.
    chunk_size : int, optional
       Chunk size for iterating through file.

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown path.
    :class:`IsADirectoryError`
       Path is a directory.
    :class:`ValueError`
       Unknown algorithm.

    Returns
    -------
    str
       Hex representation of checksum.
    """
    if algorithm not in hashlib.algorithms_guaranteed or algorithm.startswith("shake"):
        raise ValueError("Unknown algorithm")
    computed = hashlib.new(algorithm)
    if isinstance(path_or_bytes, bytes):
        computed.update(path_or_bytes)
    else:
        open_fn = gzip.open if gunzip else open
        with open_fn(path_or_bytes, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                computed.update(data)
    return computed.hexdigest()


def run_cmd_sync(cmd, cwd=None, interactive=False, timeout=None):
    """Runs a console command synchronously and returns the results.

    Parameters
    ----------
    cmd : str
       The command to execute.
    cwd : :class:`pathlib.Path`, optional
       The working directory in which to execute the command.
    interactive : bool, optional
       If set, run command interactively and pipe all output to console.
    timeout : float, optional
       If specified, kills process and throws error after this many seconds.

    Returns
    -------
    int
       Process exit status code.
    str, optional
       Standard output (if not in interactive mode).
    str, optional
       Standard error (if not in interactive mode).

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown command.
    :class:`NotADirectoryError`
       Specified working directory is not a directory.
    :class:`subprocess.TimeoutExpired`
       Specified timeout expired.
    """
    if cmd is None or len(cmd.strip()) == 0:
        raise FileNotFoundError()

    kwargs = {}
    if not interactive:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    err = None
    with subprocess.Popen(shlex.split(cmd), cwd=cwd, **kwargs) as p:
        try:
            p_res = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            err = e
        p.kill()

    if err is not None:
        raise err

    result = p.returncode

    if not interactive:
        stdout, stderr = [s.decode("utf-8").strip() for s in p_res]
        result = (result, stdout, stderr)

    return result


_RETRIEVE_AUDIO_CMD_TEMPLATE = """
youtube-dl \
    --no-cache-dir \
    --no-continue \
    --no-playlist \
    --format bestaudio/best \
    {url}
"""


def retrieve_audio_bytes(url, return_name=False, timeout=60.0):
    """Retrieves encoded audio (as raw bytes) from specified URL.

    Parameters
    ----------
    url: str
       The URL to retrieve from.
    return_name: bool
       If True, return the retrieved file name.
    timeout : float
       Max amount of time to wait before throwing an error.

    Returns
    -------
    bytes
       The raw bytes of the encoded audio file.
    str, optional
       The name of the encoded audio file, if return_name is True.

    Raises
    ------
    :class:`subprocess.TimeoutExpired`
       Specified timeout expired.
    Exception
       Error during retrieval.
    """
    with tempfile.TemporaryDirectory() as d:
        assert len(list(pathlib.Path(d).iterdir())) == 0
        status, stdout, stderr = run_cmd_sync(
            cmd=_RETRIEVE_AUDIO_CMD_TEMPLATE.format(url=url.strip()),
            cwd=d,
            timeout=timeout,
        )
        if status != 0:
            raise Exception(f"Failed to retrieve from {url}:\n{stderr}")
        paths = list(pathlib.Path(d).iterdir())
        assert len(paths) == 1
        path = paths[0]
        with open(path, "rb") as f:
            audio_bytes = f.read()
        if return_name:
            result = (audio_bytes, path.name)
        else:
            result = audio_bytes
        return result


def decode_audio(
    path_or_bytes,
    sr=None,
    offset=0.0,
    duration=None,
    mono=False,
    normalize=False,
    res_type="kaiser_best",
):
    """Decodes encoded audio from path or raw bytes.

    Parameters
    ----------
    path_or_bytes: :class:`pathlib.Path`, str, or bytes
       The filepath or raw bytes to decode.
    sr: int
       If specified, resample audio to this sample rate.
    offset: float
       Decode audio starting from this timestamp in seconds.
    duration: float
       Decode at most this many audio in seconds.
    mono: bool
       If True, average multichannel audio to mono.
    normalize: bool
       If True, normalize audio to max(abs(audio)) == 1.0.
    res_type: str
       The resampling algorithm to use (see `librosa.load` documentation).

    Returns
    -------
    int
       The sample rate of the decoded audio.
    :class:`np.ndarray`
       A NumPy array of the audio (shape [nsamps, nch], dtype float32).

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown file.
    :class:`RuntimeError`
       Unknown file format.
    """
    with tempfile.NamedTemporaryFile("wb") as f:
        # NOTE: This could be BytesIO but librosa has buggy support.
        if isinstance(path_or_bytes, bytes):
            f.write(path_or_bytes)
            f.flush()
            path = f.name
        else:
            path = path_or_bytes

        # Decode audio file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(
                    path,
                    sr=sr,
                    mono=mono,
                    offset=offset,
                    duration=duration,
                    res_type=res_type,
                )
        except audioread.exceptions.NoBackendError as e:
            raise RuntimeError("Unknown format") from e

    # Check output and rearrange to [nsamps, nch]
    assert isinstance(sr, int)
    assert audio.dtype == np.float32
    assert audio.ndim in [1, 2]
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    assert audio.shape[0] > 0
    audio = np.swapaxes(audio, 0, 1)
    audio = np.asfortranarray(audio)

    # Normalize
    if normalize and audio.shape[0] > 0:
        norm_factor = np.abs(audio).max()
        if norm_factor > 0:
            audio /= norm_factor

    return sr, audio


def encode_audio(path, sr, audio, bitexact=False, timeout=60.0):
    """Encodes raw audio array into different file formats using FFmpeg.

    Parameters
    ----------
    path: :class:`pathlib.Path`, str
       Destination filepath (where the extension determines the codec).
    sr: int
       The sample rate of the audio.
    audio: :class:`np.ndarray`
       A NumPy array of the audio (shape [nsamps, nch], dtype float32).
    bitexact: bool
       If True, ensure reproducible checksums of output file (only on FFmpeg 4+).
    timeout: float
       Maximum amount of time to wait for encoding.

    Raises
    :class:`subprocess.TimeoutExpired`
       Specified timeout expired.
    :class:`Exception`
       FFmpeg threw an error while encoding.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wavwrite(f.name, sr, audio)
        cmd = f"ffmpeg -v error -i {f.name} -y {'-bitexact' if bitexact else ''} {path}"
        status, stdout, stderr = run_cmd_sync(cmd, timeout=timeout)
        if status != 0:
            raise Exception(f"FFmpeg failed: {stderr}")
        assert pathlib.Path(path).is_file()
