"""Run a checked candidate in a process of its own.

Status: experimental

``restricted.check`` is the first barrier; this is the second. The candidate is
never imported into the caller's process. It is sent, with the arguments of
every call, to ``_jail_child.py`` started as a fresh interpreter that

* has an environment of its own making (thread counts and a fixed hash seed;
  no key, no token, no path of the caller) and a temporary working directory
  that is removed afterwards;
* cannot write a file, cannot open a new file, pipe or socket, cannot start a
  process, and is limited in memory and in CPU time;
* defines only ``np``, ``math`` and the builtins ``restricted`` allows, afresh
  for every call, with numpy's random numbers seeded the same way each time;
* gives each call its own time limit, raised as something a candidate's
  ``except Exception`` cannot catch, and ends the process if a call keeps
  computing where no exception can reach it. The calls that remain then run in
  a new process, so one bad problem costs one answer.

What comes back is one JSON value per call. The caller decides what a value
means: for a proposer, strings that the whitelist parser has yet to accept.

**Limits.** This is isolation by an operating-system process and its resource
limits, on Linux. It is not a container: the kernel is the same, and the
limits are only as good as the kernel's enforcement of them. A run with a
model that might be hostile, or on a machine that holds anything worth
stealing, should put a container or a virtual machine without network around
the whole run as well.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.evolution import restricted

CHILD = Path(__file__).with_name("_jail_child.py")
MEMORY_BYTES = 2 * 1024 ** 3
SECONDS_PER_CALL = 5.0
STARTUP_SECONDS = 20.0      # a fresh interpreter importing numpy
MAX_RESTARTS = 5            # processes started again after one died during a call
MAX_RESULT_CHARS = 64_000_000   # the most a caller may ask the child to read of one answer


@dataclass
class CallResult:
    ok: bool
    result: Any = None
    error: str = ""
    seconds: float = 0.0


@dataclass
class JailRun:
    defined: bool                              #: the candidate's code ran and defined the function
    calls: list[CallResult] = field(default_factory=list)
    error: str = ""                            #: why nothing ran
    seconds: float = 0.0
    restarts: int = 0                          #: processes that died during a call


def _encode(argument: Any) -> Any:
    if isinstance(argument, np.ndarray):
        return {"array": argument.tolist()}
    return argument


def environment() -> dict[str, str]:
    """What the jailed process is given: nothing of the caller's but, where set, the loader's library path."""
    env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]      # some interpreters cannot start without it
    return env


def _ended(returncode: int, hard: float) -> str:
    if returncode == -signal.SIGPROF:
        return f"killed after {hard:g} s of computing without returning"
    if returncode < 0:
        return f"the process was ended by signal {-returncode}"
    return f"the process ended with code {returncode}"


def _segment(code: str, function: str, calls: Sequence[Sequence[Any]], seconds_per_call: float,
             memory_bytes: int, result_chars: int | None = None) -> tuple[list[dict[str, Any]], str]:
    """One process for ``calls``: the lines it wrote, and why it ended early (or ``""``)."""
    hard = 2.0 * seconds_per_call + 1.0
    budget = STARTUP_SECONDS + 1.2 * seconds_per_call * (len(calls) + 1) + hard     # one more for defining it
    limits: dict[str, float] = {"seconds_per_call": seconds_per_call, "cpu_seconds_per_call": hard,
                                "memory_bytes": memory_bytes, "cpu_seconds": int(budget) + 5}
    if result_chars is not None:
        limits["result_chars"] = min(int(result_chars), MAX_RESULT_CHARS)
    request = {"code": code, "function": function,
               "builtins": sorted(restricted.ALLOWED_BUILTINS | restricted.ALLOWED_EXCEPTIONS),
               "attributes": sorted(restricted.ALLOWED_ATTRIBUTES),
               "calls": [[_encode(a) for a in call] for call in calls], "limits": limits}
    with tempfile.TemporaryDirectory(prefix="jail-") as directory:
        try:
            done = subprocess.run(  # noqa: S603 - this interpreter, the child script of this package, no shell
                [sys.executable, "-I", "-B", str(CHILD)], input=json.dumps(request), stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace", timeout=budget, cwd=directory,
                env=environment(), check=False)
            output, ended = done.stdout, "" if done.returncode == 0 else _ended(done.returncode, hard)
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ""
            output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            ended = f"the process ran over {budget:.0f} s and was killed"
    lines: list[dict[str, Any]] = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, RecursionError):                 # not JSON, or nested deeper than can be read
            ended = ended or "the process wrote something that is not a result"
            break
        if not isinstance(item, dict):
            ended = ended or "the process wrote something that is not a result"
            break
        lines.append(item)
    return lines, ended


def run_unchecked(code: str, function: str, calls: Sequence[Sequence[Any]], *, seconds_per_call: float = SECONDS_PER_CALL,
                  memory_bytes: int = MEMORY_BYTES, result_chars: int | None = None) -> JailRun:
    """``run`` without the syntax check: the second barrier alone. For the tests of that barrier."""
    started = time.perf_counter()
    results: list[CallResult] = []
    restarts = 0
    while True:
        lines, ended = _segment(code, function, calls[len(results):], seconds_per_call, memory_bytes, result_chars)
        if not lines or not lines[0].get("defined"):
            error = str(lines[0].get("error", "")) if lines else ""
            error = error or ended or "the process wrote nothing"
            if not results:
                return JailRun(False, [], error, time.perf_counter() - started, restarts)
            results += [CallResult(False, None, f"could not be defined again: {error}")] * (len(calls) - len(results))
            break
        results += [CallResult(bool(item.get("ok")), item.get("result"), str(item.get("error", "")),
                               float(item.get("seconds", 0.0))) for item in lines[1:][: len(calls) - len(results)]]
        if len(results) >= len(calls):
            break
        results.append(CallResult(False, None, ended or "the process ended during this call"))
        restarts += 1
        if len(results) < len(calls) and restarts > MAX_RESTARTS:
            results += [CallResult(False, None, "not run: the process had died too often")] * (len(calls) - len(results))
        if len(results) >= len(calls):
            break
    return JailRun(True, results, "", time.perf_counter() - started, restarts)


def run(code: str, function: str, calls: Sequence[Sequence[Any]], *, parameters: int | None = None,
        seconds_per_call: float = SECONDS_PER_CALL, memory_bytes: int = MEMORY_BYTES,
        result_chars: int | None = None) -> JailRun:
    """Check ``code``, then call ``function`` once per entry of ``calls`` in a jailed process.

    Raises ``restricted.RejectedCodeError`` if the code is outside the allowed subset; that is the
    one thing reported by exception, because nothing was run. Arrays among the arguments are sent
    as numbers; everything else has to be JSON. An answer longer than 200,000 characters is not
    read unless ``result_chars`` allows more (a task whose answers are whole circuits does).
    """
    restricted.check(code, function, parameters)
    return run_unchecked(code, function, calls, seconds_per_call=seconds_per_call, memory_bytes=memory_bytes,
                         result_chars=result_chars)
