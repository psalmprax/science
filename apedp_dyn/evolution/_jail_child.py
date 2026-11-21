"""The process a candidate runs in. Started by ``jail``; imports nothing of this repository.

Status: experimental

Reads one JSON request from standard input, then locks itself down: no file can be written, no
new file or socket can be opened, no new process can be started, memory and CPU time are capped.
It defines the candidate in a namespace that holds ``np``, ``math`` and the builtins the request
names, afresh for every call so that nothing carries over from one problem to the next, calls the
named function, and writes one JSON line per call to standard output. A call that raises, runs
over its time or returns something too large is reported as failed and the process goes on to the
next call. A call that keeps computing inside one numpy or integer operation, where no Python
exception can reach it, ends the process when its CPU time is up; ``jail`` then starts another for
the calls that remain.
"""
from __future__ import annotations
import __future__ as future

import builtins
import json
import math
import os
import resource
import signal
import sys
import time
from typing import Any

MAX_RESULT_CHARS = 200_000
NOBODY = 65534            # the user a jail started by root becomes


class _OutOfTime(BaseException):
    """Raised by the alarm. Not an ``Exception``, so a candidate's ``except Exception`` cannot swallow it."""


_RUNNING = [False]          # whether a candidate is running: the alarm raises only then, and only once


def _alarm(_signum: int, _frame: object) -> None:
    if _RUNNING[0]:
        _RUNNING[0] = False
        raise _OutOfTime()


def _nothing(*_args: object, **_kwargs: object) -> None:
    """What ``print`` is for a candidate."""


def main() -> int:
    request = json.loads(sys.stdin.read())
    limits = request["limits"]
    soft, hard = float(limits["seconds_per_call"]), float(limits["cpu_seconds_per_call"])
    resource.setrlimit(resource.RLIMIT_CPU, (int(limits["cpu_seconds"]), int(limits["cpu_seconds"]) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))            # no file can be written
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Everything a candidate can reach is imported before the limits on memory and on open files are
    # set: numpy loads some of its parts on first use, and after the limit nothing can be loaded.
    import linecache  # noqa: F401 - loaded now for the same reason
    import traceback  # noqa: F401
    import warnings

    import numpy as np
    import numpy.fft
    import numpy.linalg
    import numpy.ma
    import numpy.polynomial
    import numpy.random

    warnings.simplefilter("ignore")
    for holder in (np, np.linalg, np.random, np.random.default_rng(0), np.ones(1), math):
        for name in request["attributes"]:
            getattr(holder, name, None)
    resource.setrlimit(resource.RLIMIT_AS, (int(limits["memory_bytes"]), int(limits["memory_bytes"])))
    resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))           # no new file, pipe or socket can be opened
    try:
        if os.geteuid() == 0:                                    # root is exempt from the limit below: stop being root
            os.setgroups([])
            os.setgid(NOBODY)
            os.setuid(NOBODY)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))        # no new process or thread
    except (ValueError, OSError, AttributeError):                # not every platform allows it
        pass
    np.seterr(all="ignore")
    allowed: dict[str, Any] = {name: getattr(builtins, name) for name in request["builtins"] if hasattr(builtins, name)}
    if "print" in allowed:
        allowed["print"] = _nothing
    real_import = builtins.__import__

    def loaded_only(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        """What numpy's compiled code gets when it fetches one of its own helpers by name on first use.

        Such a fetch goes through the builtins of whoever is running, here the candidate, and without
        this an ordinary call (``astype``, ``np.full``, printing an array) would fail. Only a part of
        numpy that is loaded already is handed out; a candidate cannot name this function, because a
        name that starts with an underscore is refused before it runs.
        """
        if level == 0 and (name == "numpy" or name.startswith("numpy.")) and name in sys.modules:
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"{name} cannot be imported here")

    allowed["__import__"] = loaded_only
    out = sys.stdout
    try:
        compiled = compile(request["code"], "<candidate>", "exec", flags=future.annotations.compiler_flag,
                           dont_inherit=True)
    except BaseException as exc:  # noqa: BLE001 - whatever compiling the candidate raises is the report
        out.write(json.dumps({"defined": False, "error": f"{type(exc).__name__}: {exc}"[:300]}) + "\n")
        return 0
    signal.signal(signal.SIGALRM, _alarm)

    def encode(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"{type(value).__name__} cannot be returned")

    def attempt(arguments: list[Any] | None) -> str:
        """Define the candidate afresh and, with arguments, call it once; the JSON line to report."""
        started = time.perf_counter()
        try:
            try:
                signal.setitimer(signal.ITIMER_PROF, hard)       # CPU time; the default action ends the process
                signal.setitimer(signal.ITIMER_REAL, soft)
                _RUNNING[0] = True
                try:
                    namespace: dict[str, Any] = {"__builtins__": dict(allowed), "np": np, "math": math}
                    np.random.seed(0)
                    exec(compiled, namespace)  # nosec B102  # noqa: S102 - the checked candidate, in its jail
                    function = namespace[request["function"]]
                    if arguments is None:
                        return json.dumps({"defined": True})
                    line = json.dumps({"ok": True, "result": function(*arguments)}, default=encode)
                finally:
                    _RUNNING[0] = False                          # from here on the alarm raises nothing
            except _OutOfTime:
                return json.dumps({"ok": False, "defined": False, "error": f"no answer within {soft:g} s"})
            except BaseException as exc:  # noqa: BLE001 - whatever the candidate raises is the report for this call
                return json.dumps({"ok": False, "defined": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.setitimer(signal.ITIMER_PROF, 0.0)
        seconds = time.perf_counter() - started
        if seconds > soft:
            return json.dumps({"ok": False, "error": f"answered after {seconds:.1f} s; the limit is {soft:g} s"})
        if len(line) > MAX_RESULT_CHARS:
            return json.dumps({"ok": False, "error": f"the answer has {len(line)} characters; at most "
                                                     f"{MAX_RESULT_CHARS} are read"})
        return line[:-1] + f', "seconds": {seconds:.4f}}}'

    first = attempt(None)
    out.write(first + "\n")
    out.flush()
    if not json.loads(first).get("defined"):
        return 0
    for call in request["calls"]:
        arguments = [np.asarray(a["array"], dtype=float) if isinstance(a, dict) and "array" in a else a for a in call]
        out.write(attempt(arguments) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - a fault of the jail itself is reported as one, then the process ends
        sys.stdout.write(json.dumps({"defined": False, "error": f"jail: {type(exc).__name__}: {exc}"[:300]}) + "\n")
        sys.exit(0)
