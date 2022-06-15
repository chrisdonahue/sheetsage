import json
import logging
import pathlib
import urllib.request

from . import CACHE_DIR, LIB_DIR
from .utils import compute_checksum

_DEFAULT_CHUNK_SIZE = 4096
_ASSETS = None

def _init_assets():
    global _ASSETS
    if _ASSETS is not None:
        raise Exception("Should only run this once")

    _ASSETS = {}
    asset_paths = set()
    for json_path in sorted(pathlib.Path(LIB_DIR, "assets").rglob("*.json")):
        with open(json_path, "r") as f:
            d = json.load(f)
        for tag, asset in d.items():
            if "checksum" not in asset:
                raise AssertionError("Missing checksum")
            try:
                asset["path"] = pathlib.PurePosixPath(asset["path"].strip())
            except:
                raise AssertionError("Invalid path")
            if asset["path"] in asset_paths:
                raise AssertionError("Duplicate path")
            asset_paths.add(asset["path"])
            asset["path_abs"] = pathlib.Path(CACHE_DIR, asset["path"])
        _ASSETS.update(d)


_init_assets()


def get_asset_tags():
    return set(_ASSETS.keys())


def _download(url, dest_path, chunk_size=_DEFAULT_CHUNK_SIZE):
    with open(dest_path, "wb") as f:
        r = urllib.request.urlopen(url)
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def retrieve_asset(tag, delete_wrong=False, chunk_size=_DEFAULT_CHUNK_SIZE):
    """Attempts to acquire and/or verify existance of a tagged asset in the cache.

    Returns
    -------
    str
       Absolute file path for asset, if verified.

    Raises
    ------
    :class:`ValueError`
       Invalid asset tag.
    :class:`Exception`
       Asset could not be verified.
    """
    # Retrieve asset
    if tag not in _ASSETS:
        raise ValueError()
    asset = _ASSETS[tag]
    path = asset["path_abs"]
    checksum = asset["checksum"]
    logging.info(f"Verifying asset: {tag}")
    logging.info(f"Asset location: {path}")

    # Create parent directory
    if not path.parent.is_dir():
        logging.info(f"Creating parent: {path.parent}")
        path.parent.mkdir(parents=True)

    def verify():
        assert path.is_file()
        if checksum is not None:
            if len(checksum) == 32:
                algorithm = "md5"
            elif len(checksum) == 40:
                algorithm = "sha1"
            elif len(checksum) == 64:
                algorithm = "sha256"
            else:
                raise AssertionError("Unknown checksum algorithm")
            computed = compute_checksum(
                path, algorithm=algorithm, chunk_size=chunk_size
            )
            if computed != checksum:
                raise Exception(f"File {path} has wrong checksum.")

    # Delete incorrect files
    already_verified = False
    if delete_wrong and path.is_file():
        try:
            verify()
            already_verified = True
        except Exception:
            logging.warning(f"Deleting file with bad checksum: {path}")
            path.unlink()

    # Attempt to download
    if not path.is_file():
        url = asset.get("url")
        if url is None:
            raise Exception("File is missing and cannot be downloaded")
        logging.info(f"Downloading from: {url}")
        try:
            _download(url, path)
        except Exception as e:
            if path.is_file():
                path.unlink()
            raise Exception(f"Download failed: {e}")
    assert path.is_file()

    # Ensure file integrity
    if not already_verified:
        verify()
    logging.info(f"Verified!")

    return path


if __name__ == "__main__":
    import multiprocessing
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("startswith", nargs="?")
    parser.add_argument("--delete_wrong", action="store_true", dest="delete_wrong")
    parser.add_argument("--num_parallel", "-n", type=int)

    parser.set_defaults(startswith=None, num_parallel=1, delete_wrong=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tags = get_asset_tags()
    if args.startswith is not None:
        tags = [t for t in tags if t.startswith(args.startswith.strip().upper())]

    def task(t):
        logging.info("-" * 80)
        try:
            retrieve_asset(t, delete_wrong=args.delete_wrong)
        except Exception as e:
            logging.error(e)
            raise e

    with multiprocessing.Pool(args.num_parallel) as p:
        p.map(task, tags)
