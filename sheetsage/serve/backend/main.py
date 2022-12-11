import json
import multiprocessing
import pathlib
import traceback
from enum import Enum

from flask import Flask, abort, jsonify, request, send_file

from ...infer import Status as SheetSageStatus
from ...infer import sheetsage
from ...utils import compute_checksum, retrieve_audio_bytes

APP = Flask(__name__)


class JobStatus(Enum):
    QUEUED = 0
    FETCHING = 1
    RUNNING = 2
    FINALIZED = 3


class JobError(Exception):
    pass


class FetchAudioError(JobError):
    pass


class BulkyAudioError(JobError):
    pass


_MANAGER = multiprocessing.Manager()
_JOB_QUEUE = _MANAGER.Queue()
_JOB_INPUTS = _MANAGER.dict()
_JOB_STATUS = _MANAGER.dict()
_JOB_OUTPUTS = _MANAGER.dict()


def _work(wid):
    while True:
        print(f"(WID {wid}) Waiting for job")
        jid = _JOB_QUEUE.get()
        job_def = _JOB_INPUTS[jid]

        print(f"(WID {wid}) Working on {jid}:\n{job_def}")

        def status_change_callback(s):
            print(f"(WID {wid}) Status update for {jid}: {s.name}")
            assert isinstance(s, JobStatus) or isinstance(s, SheetSageStatus)
            _JOB_STATUS[jid] = s

        output = None
        stack_trace = None

        # Fetch audio
        if isinstance(job_def["audio_path_bytes_or_url"], str):
            status_change_callback(JobStatus.FETCHING)
            try:
                audio_bytes = retrieve_audio_bytes(
                    job_def["audio_path_bytes_or_url"],
                    max_filesize_mb=ARGS["fetch_max_filesize_mb"],
                    max_duration_seconds=ARGS["fetch_max_duration_seconds"],
                    timeout=ARGS["fetch_timeout_seconds"],
                )
                job_def["audio_path_bytes_or_url"] = audio_bytes
            except ValueError:
                output = BulkyAudioError()
                stack_trace = traceback.format_exc()
            except Exception:
                output = FetchAudioError()
                stack_trace = traceback.format_exc()

        # Run
        if stack_trace is None:
            status_change_callback(JobStatus.RUNNING)
            try:
                lead_sheet, segment_beats, segment_beats_times = sheetsage(
                    **job_def, status_change_callback=status_change_callback
                )
                output_path = pathlib.Path(ARGS["tmp_dir"], f"{jid}.json")
                with open(output_path, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "lead_sheet": lead_sheet,
                                "segment_beats": segment_beats,
                                "segment_beats_times": segment_beats_times,
                            }
                        )
                    )
                output = output_path
            except Exception as e:
                output = JobError()
                stack_trace = traceback.format_exc()

        # Finalize
        print(f"(WID {wid}) Finalizing {jid}")
        assert isinstance(output, pathlib.Path) or isinstance(output, JobError)
        _JOB_OUTPUTS[jid] = output
        if isinstance(output, pathlib.Path):
            status_change_callback(JobStatus.FINALIZED)
        else:
            assert stack_trace is not None
            print(f"(WID {wid}) Exception during {jid}:\n{stack_trace.strip()}")


@APP.errorhandler(400)
@APP.errorhandler(500)
def _api_error(e):
    return jsonify(e.description), e.code


@APP.route("/ping", methods=["GET"])
def ping():
    return "Pong", 200


@APP.route("/submit", methods=["POST"])
def submit():
    # Check payload size
    if ARGS["max_payload_size_mb"] is not None and request.content_length > (
        ARGS["max_payload_size_mb"] * 1024 * 1024
    ):
        abort(413, description="Too large")

    # Define arguments
    arg_to_sanitize_fn = {
        "audio_url": str,
        "audio_file": None,
        "segment_start_hint": float,
        "segment_end_hint": float,
        "legacy_behavior": lambda i: bool(int(i)),
    }

    # Check arguments
    if request.json is not None:
        r = dict(request.json)
    elif request.form is not None:
        r = dict(request.form)
    else:
        abort(400, description="Unknown request format")
    for k in r.keys():
        if k not in arg_to_sanitize_fn:
            abort(400, description=f"Unknown argument: {k}")

    # Sanitize arguments
    for k, fn in arg_to_sanitize_fn.items():
        if k in r and fn is not None:
            try:
                r[k] = fn(r[k])
            except:
                abort(400, description=f"Bad '{k}'")

    # Create job definition
    job_def = {
        "audio_path_bytes_or_url": None,
        "segment_start_hint": None,
        "segment_end_hint": None,
        "use_jukebox": ARGS["jukebox"],
        "legacy_behavior": False,
    }

    # Parse audio_url and audio_file
    audio_file = request.files.get("audio_file")
    if audio_file is not None:
        # Audio was uploaded
        try:
            audio_mimetype = audio_file.content_type
            audio_file_bytes = BytesIO()
            audio_file.save(audio_file_bytes)
            audio_file_bytes.seek(0)
            audio_file_bytes = audio_file_bytes.read()
            audio_file_checksum = compute_checksum(audio_file_bytes, algorithm="sha256")
        except:
            abort(400, description="Bad 'audio_file'")
        try:
            audio_path = pathlib.Path(ARGS["tmp_dir"], "audio", audio_file_checksum)
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            if not audio_path.is_file():
                with open(audio_path, "wb") as f:
                    f.write(audio_file_bytes)
        except:
            abort(500)
        job_def["audio_path_bytes_or_url"] = audio_path
    elif "audio_url" in r:
        # Media needs to be retrieved from URL
        try:
            audio_url = r["audio_url"].strip()
            assert len(audio_url) > 0
        except:
            abort(400, description="Bad 'audio_url'")
        job_def["audio_path_bytes_or_url"] = audio_url
    else:
        abort(400, description="No audio specified")

    # Parse float args
    for k in ["segment_start_hint", "segment_end_hint", "legacy_behavior"]:
        if k in r:
            job_def[k] = r[k]

    # Compute job ID
    jid = compute_checksum(
        json.dumps(job_def, sort_keys=True, indent=2).encode("utf-8"),
        algorithm="sha1",
    )

    # Submit to queue
    position = None
    status = _JOB_STATUS.get(jid)
    output = _JOB_OUTPUTS.get(jid)
    actively_processing = output is None and status is not None
    already_cached = isinstance(output, pathlib.Path) and output.is_file()
    if not (actively_processing or already_cached):
        position = _JOB_QUEUE.qsize()
        _JOB_INPUTS[jid] = job_def
        _JOB_STATUS[jid] = JobStatus.QUEUED
        if output is not None:
            del _JOB_OUTPUTS[jid]
        _JOB_QUEUE.put(jid)

    return {"jid": jid, "cached": already_cached, "position": position}


@APP.route("/heartbeat/<jid>", methods=["GET"])
def heartbeat(jid):
    status = _JOB_STATUS.get(jid)
    if status is None:
        abort(404, description="INVALID_ID")
    output = _JOB_OUTPUTS.get(jid)
    if isinstance(output, BulkyAudioError):
        abort(400, description="AUDIO_TOO_LONG_OR_TOO_BIG")
    elif isinstance(output, JobError):
        abort(500, description=status.name)
    return jsonify(status.name)


@APP.route("/lead-sheet/<jid>", methods=["GET"])
def download(jid):
    if isinstance(jid, str) and jid.endswith(".json"):
        jid = jid[:-5]
    output = _JOB_OUTPUTS.get(jid)
    if output is None:
        abort(404, description="INVALID_ID")
    if not isinstance(output, pathlib.Path):
        abort(500)
    return send_file(output, download_name=f"{jid}.json", max_age=7 * 24 * 60 * 60)


def __init():
    import os
    from argparse import ArgumentParser

    from flask_cors import CORS

    parser = ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--cors", action="store_true")
    parser.add_argument("--cors_allow", type=str)
    parser.add_argument("--jukebox", action="store_true")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_payload_size_mb", type=int)
    parser.add_argument("--fetch_max_filesize_mb", type=int)
    parser.add_argument("--fetch_max_duration_seconds", type=float)
    parser.add_argument("--fetch_timeout_seconds", type=int)
    parser.add_argument("--tmp_dir", type=str)
    parser.set_defaults(
        port=8000,
        cors=False,
        cors_allow=None,
        jukebox=False,
        num_workers=1,
        max_payload_size_mb=32,
        fetch_max_filesize_mb=128,
        fetch_max_duration_seconds=660,
        fetch_timeout_seconds=60,
        tmp_dir="/tmp/sheetsage",
    )

    global ARGS
    ARGS = vars(parser.parse_args())
    print(ARGS)

    # Indicate that Jukebox support is forthcoming
    if ARGS["jukebox"] and ARGS["num_workers"] > 1:
        raise NotImplementedError()

    # Enable CORS
    if ARGS["cors"] or ARGS["cors_allow"] is not None:
        kwargs = {}
        if ARGS["cors_allow"] is not None:
            kwargs["origins"] = [o.strip() for o in ARGS["cors_allow"].split(",")]
        CORS(APP, **kwargs)

    # Create tmp dir
    ARGS["tmp_dir"] = pathlib.Path(ARGS["tmp_dir"])
    ARGS["tmp_dir"].mkdir(parents=True, exist_ok=True)

    # Worker processes
    if ARGS["num_workers"] <= 0:
        raise ValueError()
    processes = [
        multiprocessing.Process(target=_work, args=(wid,))
        for wid in range(ARGS["num_workers"])
    ]
    [p.start() for p in processes]

    # Start HTTP server
    gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")
    if not gunicorn:
        APP.run(debug=True, use_reloader=True, host="0.0.0.0", port=ARGS["port"])

    # Join workers
    [p.join() for p in processes]


if __name__ == "__main__":
    __init()
