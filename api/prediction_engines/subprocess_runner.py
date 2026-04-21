# api/prediction_engines/subprocess_runner.py
#
# Shared helper for running prediction subprocesses and streaming their output.
#
# All prediction engines use this helper so that progress reporting and OOM
# detection are handled consistently in one place.

import subprocess
from api.models import Job
from api.services.embedding_progress_service import (
    start_embedding_tracking,
    stop_embedding_tracking,
)
from api.services.job_progress_service import set_stage_prediction_progress


def run_prediction_subprocess(
    command: list[str],
    job: Job,
    env: dict | None = None,
    label: str = "subprocess",
    method_key: str | None = None,
    target: str | None = None,
    valid_sequences: list[str] | None = None,
) -> None:
    """
    Run a prediction subprocess, stream its stdout line-by-line, and update
    the job's progress fields whenever the script emits a progress line.

    Expected progress line format (written by prediction scripts)::

        Progress: <done>/<total>

    For example: ``Progress: 42/100``

    Parameters
    ----------
    command : list[str]
        Full command to execute, including the python interpreter path and
        all arguments.
    job : Job
        Django Job instance.  Its ``predictions_made`` and
        ``total_predictions`` fields are updated in real time.
    env : dict | None
        Environment variables passed to the subprocess.  Defaults to the
        current process environment if None.
    label : str
        Short label used in log output to identify which engine is running
        (e.g. ``"DLKcat"``).

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess exits with a non-zero return code.
    """
    tracking_started = False
    if method_key and target and valid_sequences:
        tracking_started = start_embedding_tracking(
            job_public_id=job.public_id,
            method_key=method_key,
            target=target,
            valid_sequences=valid_sequences,
            env=env,
        )

    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                print(f"[{label}]", line)

                if line.startswith("Progress:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            done, total = parts[1].split("/")
                            done_i = int(done)
                            total_i = int(total)
                            if method_key and target:
                                set_stage_prediction_progress(
                                    job_public_id=job.public_id,
                                    target=target,
                                    method_key=method_key,
                                    done=done_i,
                                    total=total_i,
                                )
                            else:
                                job.predictions_made = done_i
                                job.total_predictions = total_i
                                job.save(update_fields=["predictions_made", "total_predictions"])
                        except (ValueError, AttributeError):
                            pass  # malformed progress line — ignore

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        if tracking_started:
            final_state = "done" if (process is not None and process.returncode == 0) else "error"
            stop_embedding_tracking(job.public_id, final_state=final_state)
