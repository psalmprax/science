"""The three levels: finding laws, improving the finder, improving how the finder is improved.

Status: experimental

**Level 0** (``solve``). One problem through the route in force
(``learning_loop.discover_law``: first pass, referee, last checks, measured
range), with the searcher's memory and its library's candidates beside the
first pass, and the generator's code if it has one. ``judge.verdict`` says
whether it was solved.

**Level 1** (``generation``, ``run_level1``). One generation:

1. *study*: a batch from the curriculum at the current level. A law is added
   to what the searcher has learned if the route reported it, the learning
   gate allows it (``learning_loop.learnable``) and the judge counts it solved:
   accurate on clean rows and no longer than the true law. The last condition
   uses the true law, which a self-made curriculum has and a table from the
   world does not: outside a curriculum the first two are all there is. It
   keeps accurate approximations out of what is learned: in the first run they
   were learned, and one cubic in the memory was enough for it to offer ratios
   of cubics on every later problem. The memory is rebuilt from the laws
   learned and the library re-chosen by compression; the level moves;
2. *vary the settings*: ``meta.variants`` copies of the searcher with settings
   moved by ``meta.step``. The incumbent and the copies run on one batch drawn
   for that comparison. A copy replaces the incumbent only if it reports no
   more wrong laws and either solves more at a higher rate per CPU-hour or
   solves as many for clearly less CPU (``better``);
3. *vary the code* (only with a coding model the owner provided): the loop of
   ``apedp_dyn.evolution`` rewrites a candidate generator against solved study
   problems as teaching and fresh problems as score; its champion is adopted
   under the same rule as a settings copy. Without a model this step records
   that it did not run;
4. *measure*: the searcher that came out runs on problems drawn for the
   measurement alone, from the fixed distribution. That gives ``g_k``.

Selection and measurement never share a problem: the batch that chose a
variant flatters it, so its rate there is not reported as ``g``.

**Level 2** (``run_level2``). ``Meta`` is how Level 1 works: how many variants,
how far they move, how many problems study, decide and measure, and whether
unsolved study problems are tried again. Several ``Meta`` each run Level 1
from the same starting searcher on their own problems; each is scored by the
rise of ``g`` from its first generation to its last per CPU-hour it spent in
all. The best becomes the incumbent of the next round, where it is run again
on new problems beside new variants of itself, so a ``Meta`` chosen by luck
has to win twice.

**Fixed throughout**: ``judge.seal`` is taken at the start and checked at every
generation, and every searcher's settings pass ``judge.check_settings``.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from apedp_dyn.conf.discovery_checks import DISCOVERY_ACCURACY_R2
from apedp_dyn.discovery_l2 import learning_loop
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.proposers.composed import CompositeProposer
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.structure_memory import Memory
from apedp_dyn.recursion import judge
from apedp_dyn.recursion.curriculum import Curriculum, Problem
from apedp_dyn.recursion.library import Library, LibraryProposer
from apedp_dyn.recursion.progress import Measurement, verdict

TARGET = "__target__"
DEFAULT_SETTINGS: dict[str, float | int] = {"search_seconds": 0.0, "memory_seconds": 10.0, "library_seconds": 10.0,
                                            "library_blocks": 16, "library_top": 6, "library_proposals": 3}
MAX_PROPOSALS = 10      # candidates of a stage the path confirms, as in the measured search stage
NO_MODEL = "no coding model was provided; the code of the generator was not varied"


@dataclass
class Searcher:
    """Everything Level 1 may change: the settings, what was learned, and (with a coding model) a generator's code."""

    settings: dict[str, float | int] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    laws: list[str] = field(default_factory=list)          #: laws learned, in order; the memory and library are built from them
    code: str | None = None

    def __post_init__(self) -> None:
        self.settings = judge.check_settings({**DEFAULT_SETTINGS, **self.settings})

    def library(self) -> Library:
        return Library.learn(self.laws, max_blocks=int(self.settings["library_blocks"]))

    def state(self) -> learning_loop.LearningState:
        state = learning_loop.LearningState(Memory(), [])
        for law in self.laws:
            state.learn(law, source="recursion")
        return state

    def proposer(self, seed: int = 0) -> ProposerBase | None:
        """What stands beside the first pass and the memory: the route's searches, the library, a generator's code.

        The searches are the stage in force (``search_stage.default_search_stage``: four different searches where four
        cores are free, else one), for ``search_seconds``, built by ``hard_exit.search_stage_that_cannot_hang``: the
        same members, seeds and budget, with search processes that leave without Julia's exit hook. With the frozen
        stage every job of the first run with the searches stopped for ever after one to seven searches (2026-11-24).
        Members answer in turn, each one's best first.
        """
        parts: list[ProposerBase] = []
        if self.settings["search_seconds"] > 0:
            from apedp_dyn.proposers.hard_exit import search_stage_that_cannot_hang

            parts.append(search_stage_that_cannot_hang(float(self.settings["search_seconds"]), seed=seed)[0])
        if self.settings["library_seconds"] > 0 and self.settings["library_blocks"] > 0 and self.laws:
            parts.append(FitGuardProposer(LibraryProposer(
                self.library(), max_proposals=int(self.settings["library_proposals"]),
                seconds=float(self.settings["library_seconds"]), top=int(self.settings["library_top"]), target_col=TARGET),
                target_col=TARGET))
        if self.code:
            from apedp_dyn.evolution.evolved_proposer import EvolvedProposer
            parts.append(FitGuardProposer(EvolvedProposer(self.code, target_col=TARGET), target_col=TARGET))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else CompositeProposer(parts, max_proposals=MAX_PROPOSALS)

    def as_dict(self) -> dict[str, Any]:
        return {"settings": dict(self.settings), "laws": list(self.laws), "code": self.code}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Searcher:
        return cls(dict(record.get("settings", {})), list(record.get("laws", [])), record.get("code"))


def _cpu() -> float:
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def solve(searcher: Searcher, problem: Problem) -> dict[str, Any]:
    """Level 0: one problem through the route, learning nothing; the judge's verdict and the CPU it took."""
    rows = problem.rows()
    names = problem.names
    seed = int.from_bytes(hashlib.sha256(problem.key.encode()).digest()[:4], "big")
    started = _cpu()
    try:
        record = learning_loop.discover_law(
            {n: rows["X"][:, i] for i, n in enumerate(names)}, rows["y"], rows["pool_X"], lambda idx: rows["pool_y"][idx],
            state=searcher.state(), proposer=searcher.proposer(seed), learn=False, source=problem.key, seed=seed,
            memory_seconds=float(searcher.settings["memory_seconds"]))
    except Exception as exc:  # noqa: BLE001 - a problem the route cannot run is counted unsolved, with its error
        return {"key": problem.key, "level": problem.level, "noise": problem.noise, "reported": False, "solved": False,
                "wrong": False, "learnable": False, "law": None, "cpu_seconds": round(_cpu() - started, 3),
                "error": f"{type(exc).__name__}: {exc}"[:200]}
    out = judge.verdict(record, names, problem.truth, rows["test_X"], rows["test_clean"])
    out.update({"key": problem.key, "level": problem.level, "noise": problem.noise,
                "learnable": learning_loop.learnable(record) is None, "cpu_seconds": round(_cpu() - started, 3)})
    return out


def _job(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    searcher, problem = job
    return solve(Searcher.from_dict(searcher), Problem(**problem))


def run_batch(searcher: Searcher, problems: Sequence[Problem], workers: int = 1) -> list[dict[str, Any]]:
    """Every problem with the same frozen searcher, in ``workers`` processes."""
    jobs = [(searcher.as_dict(), p.as_dict()) for p in problems]
    if workers <= 1 or len(jobs) <= 1:
        return [_job(job) for job in jobs]
    with mp.get_context("fork").Pool(min(workers, len(jobs)), maxtasksperchild=4) as pool:
        return pool.map(_job, jobs, chunksize=1)


def tally(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cpu = float(sum(r["cpu_seconds"] for r in rows))
    solved = sum(r["solved"] for r in rows)
    return {"problems": len(rows), "solved": solved, "accurate": sum(r.get("accurate", r["solved"]) for r in rows),
            "wrong": sum(r["wrong"] for r in rows), "errors": sum("error" in r for r in rows),
            "reported": sum(r["reported"] for r in rows), "cpu_seconds": round(cpu, 2),
            "per_cpu_hour": round(solved / (cpu / 3600.0), 3) if cpu > 0 else 0.0}


@dataclass(frozen=True)
class Meta:
    """How Level 1 works: what Level 2 varies."""

    variants: int = 2            #: settings copies tried per generation
    step: float = 0.5            #: how far a copy moves a setting (a share of its value)
    study: int = 12              #: problems studied per generation
    selection: int = 10          #: problems that decide between the incumbent and its copies
    retry: bool = False          #: put the last generation's unsolved study problems back into the next study batch

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def vary_settings(settings: dict[str, float | int], step: float, rng: np.random.Generator) -> dict[str, float | int]:
    """A copy with one or two settings moved up or down by ``step`` of their value, kept inside the judge's bounds."""
    out = dict(settings)
    names = sorted(judge.SETTINGS)
    for name in rng.choice(names, size=int(rng.integers(1, 3)), replace=False):
        low, high, whole = judge.SETTINGS[str(name)]
        value = float(out[str(name)])
        moved = value * (1 + step) if rng.random() < 0.5 else value * (1 - step)
        if whole:
            moved = round(moved) if round(moved) != value else value + (1 if moved >= value else -1)
        out[str(name)] = int(np.clip(moved, low, high)) if whole else float(np.clip(round(moved, 2), low, high))
    return judge.check_settings(out)


CPU_MARGIN = 0.2        # with as many solved, a copy has to save this share of the CPU time to be taken


def better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """No more wrong laws, and either more solved at a higher rate, or as many solved for clearly less CPU.

    The first run of the levels took a copy whenever its rate was higher at all. With the same problems solved, a
    copy that happened to use 2 % less processor time replaced the incumbent, so the settings drifted on timing
    noise (seven of eight copies taken, none having solved more than one extra problem). ``CPU_MARGIN`` is a
    development constant of this rule.
    """
    if candidate["wrong"] > incumbent["wrong"]:
        return False
    if candidate["solved"] > incumbent["solved"]:
        return bool(candidate["per_cpu_hour"] > incumbent["per_cpu_hour"])
    if candidate["solved"] == incumbent["solved"]:
        return bool(candidate["cpu_seconds"] < (1 - CPU_MARGIN) * incumbent["cpu_seconds"])
    return False


def code_step(searcher: Searcher, client: Callable[[str], str] | None, teaching: Sequence[Problem],
              scored: Sequence[Problem], *, steps: int = 6, seconds: float = 600.0) -> tuple[str | None, dict[str, Any]]:
    """A generator's code varied by a coding model against a fixed judge; the champion's code, or ``None``."""
    if client is None:
        return None, {"ran": False, "reason": NO_MODEL}
    import sympy as sp

    from apedp_dyn.evolution import operator as loop
    from apedp_dyn.evolution.evolved_proposer import CandidateGeneratorTask
    from apedp_dyn.evolution.evolved_proposer import Problem as GeneratorProblem
    from apedp_dyn.evolution.seed_generator import SEED_GENERATOR

    def as_task_problem(problem: Problem) -> GeneratorProblem:
        rows = problem.rows()
        return GeneratorProblem(problem.key, rows["X"], rows["y"], rows["test_X"], problem.truth)

    def recovered(expr: sp.Expr, problem: GeneratorProblem) -> bool:
        symbols = [sp.Symbol(n) for n in problem.names]
        with np.errstate(all="ignore"):
            pred = np.asarray(sp.lambdify(symbols, expr, "numpy")(*problem.X_test.T), dtype=float) * np.ones(len(problem.X_test))
            clean = np.asarray(sp.lambdify(symbols, problem.truth, "numpy")(*problem.X_test.T), dtype=float)
        r2 = judge.r2(clean * np.ones(len(problem.X_test)), pred)
        return bool(r2 is not None and r2 >= DISCOVERY_ACCURACY_R2)

    task = CandidateGeneratorTask([as_task_problem(p) for p in scored], recovered,
                                  teaching=[as_task_problem(p) for p in teaching])
    archive = loop.evolve(client, task, searcher.code or SEED_GENERATOR, steps=steps, seconds=seconds)
    best = archive.best()
    report = {"ran": True, **loop.summary(archive)}
    if best is None or best.origin == "seed" or not best.evaluation.ran:
        return None, report
    return best.code, report


def improve(searcher: Searcher, curriculum: Curriculum, meta: Meta, k: int, *, workers: int = 1,
            client: Callable[[str], str] | None = None, carried: Sequence[Problem] = (), sealed: dict[str, str] | None = None,
            rng: np.random.Generator | None = None) -> tuple[Searcher, dict[str, Any], list[Problem]]:
    """Steps 1 to 3 of a generation: study, vary the settings, vary the code. No measurement.

    Returns the searcher that came out, the record, and the study problems left unsolved.
    """
    if sealed is not None:
        judge.check(sealed)
    rng = rng or np.random.default_rng(k)
    spent = 0.0
    # 1. study
    again = list(carried)[: meta.study // 2] if meta.retry else []
    study = [*again, *curriculum.batch(meta.study - len(again), f"study {k}")]
    rows = run_batch(searcher, study, workers)
    spent += sum(r["cpu_seconds"] for r in rows)
    fresh = rows[len(again):]
    curriculum.update(sum(r["solved"] for r in fresh), len(fresh), f"study {k}")
    learned = [str(r["law"]) for r in rows if r["solved"] and r["learnable"] and r["law"] and r["law"] not in searcher.laws]
    unsolved = [p for p, r in zip(study, rows, strict=True) if not r["solved"]]
    grown = replace(searcher, laws=[*searcher.laws, *learned])
    record: dict[str, Any] = {"generation": k, "study": {**tally(rows), "retried": len(again), "learned": len(learned),
                                                         "level": curriculum.level},
                              "laws_known": len(grown.laws), "library_blocks": len(grown.library().blocks)}
    # 2. vary the settings
    deciding = curriculum.measurement(meta.selection, f"selection {k}")
    incumbent_rows = run_batch(grown, deciding, workers)
    spent += sum(r["cpu_seconds"] for r in incumbent_rows)
    incumbent, chosen = tally(incumbent_rows), grown
    tried = []
    for _ in range(meta.variants):
        moved = vary_settings(dict(grown.settings), meta.step, rng)
        for _again in range(20):                       # a copy equal to the incumbent would be run for nothing
            if moved != grown.settings:
                break
            moved = vary_settings(dict(grown.settings), meta.step, rng)
        copy = replace(grown, settings=moved)
        copy_rows = run_batch(copy, deciding, workers)
        spent += sum(r["cpu_seconds"] for r in copy_rows)
        score = tally(copy_rows)
        adopted = better(score, incumbent)
        tried.append({"settings": dict(copy.settings), **score, "adopted": adopted})
        if adopted:
            incumbent, chosen = score, copy
    record["selection"] = {"incumbent": tally(incumbent_rows), "variants": tried, "settings": dict(chosen.settings)}
    # 3. vary the code
    solved_study = [p for p, r in zip(study, rows, strict=True) if r["solved"]]
    code, report = code_step(chosen, client, solved_study, curriculum.measurement(meta.selection, f"code {k}")
                             if client is not None else [])
    if code is not None:
        with_code = replace(chosen, code=code)
        code_rows = run_batch(with_code, deciding, workers)
        spent += sum(r["cpu_seconds"] for r in code_rows)
        score = tally(code_rows)
        report["selection"] = {**score, "adopted": better(score, incumbent)}
        if report["selection"]["adopted"]:
            chosen = with_code
    record["code"] = report
    record["cpu_seconds_improving"] = round(spent, 2)
    return chosen, record, unsolved


def measurement_of(k: int, rows: Sequence[dict[str, Any]]) -> Measurement:
    counts = tally(rows)
    return Measurement(k, counts["problems"], counts["solved"], counts["wrong"], counts["reported"], counts["cpu_seconds"],
                       counts["accurate"])


def generation(searcher: Searcher, curriculum: Curriculum, meta: Meta, k: int, *, measure: int, workers: int = 1,
               client: Callable[[str], str] | None = None, carried: Sequence[Problem] = (), sealed: dict[str, str] | None = None,
               rng: np.random.Generator | None = None) -> tuple[Searcher, dict[str, Any], list[Problem]]:
    """One generation of Level 1: ``improve``, then the measurement on problems drawn for it alone."""
    chosen, record, unsolved = improve(searcher, curriculum, meta, k, workers=workers, client=client, carried=carried,
                                       sealed=sealed, rng=rng)
    if sealed is not None:
        judge.check(sealed)
    measured = run_batch(chosen, curriculum.measurement(measure, f"measurement {k}"), workers)
    record["measurement"] = measurement_of(k, measured).as_dict()
    record["rows"] = {"measurement": measured}
    return chosen, record, unsolved


def run_level1(start: Searcher, *, generations: int, meta: Meta, seed: int, measure: int = 20, workers: int = 1,
               client: Callable[[str], str] | None = None, tag: str = "",
               on_generation: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Level 1 for ``generations`` generations after a measurement of the starting searcher (generation 0)."""
    sealed = judge.seal()
    curriculum = Curriculum(int.from_bytes(hashlib.sha256(f"{seed}|{tag}".encode()).digest()[:6], "big"))
    rng = np.random.default_rng(curriculum.seed)
    started = time.perf_counter()
    first = tally(run_batch(start, curriculum.measurement(measure, "measurement 0"), workers))
    measurements = [Measurement(0, first["problems"], first["solved"], first["wrong"], first["reported"], first["cpu_seconds"],
                                first["accurate"])]
    if on_generation is not None:
        on_generation({"generation": 0, "measurement": measurements[0].as_dict()})
    carried: list[Problem] = []
    searcher, records = start, []
    improving = 0.0
    for k in range(1, generations + 1):
        searcher, record, carried = generation(searcher, curriculum, meta, k, measure=measure, workers=workers, client=client,
                                               carried=carried, sealed=sealed, rng=rng)
        improving += record["cpu_seconds_improving"]
        m = record["measurement"]
        measurements.append(Measurement(k, m["problems"], m["solved"], m["wrong"], m["reported"], m["cpu_seconds"], m["accurate"]))
        records.append(record)
        if on_generation is not None:
            on_generation({k_: v for k_, v in record.items() if k_ != "rows"})
    judge.check(sealed)
    total = improving + sum(m.cpu_seconds for m in measurements)
    gain = measurements[-1].g - measurements[0].g
    return {"meta": meta.as_dict(), "seed": seed, "tag": tag, "generation_0": measurements[0].as_dict(), "generations": records,
            "measurements": [m.as_dict() for m in measurements], "progress": verdict(measurements),
            "searcher": searcher.as_dict(), "curriculum": curriculum.history, "sealed_files": len(sealed),
            "cpu_seconds": round(total, 1), "seconds": round(time.perf_counter() - started, 1),
            "gain_per_cpu_hour": round(gain / (total / 3600.0), 4) if total > 0 else 0.0}


def vary_meta(meta: Meta, rng: np.random.Generator) -> Meta:
    """A copy of how Level 1 works with one thing changed."""
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return replace(meta, variants=int(np.clip(meta.variants + rng.choice([-1, 1]), 1, 6)))
    if kind == 1:
        return replace(meta, step=float(np.clip(round(meta.step * rng.choice([0.5, 2.0]), 3), 0.05, 0.9)))
    if kind == 2:
        return replace(meta, study=int(np.clip(round(meta.study * rng.choice([0.5, 1.5])), 4, 60)))
    if kind == 3:
        return replace(meta, selection=int(np.clip(round(meta.selection * rng.choice([0.5, 1.5])), 4, 60)))
    return replace(meta, retry=not meta.retry)


def metas_for_round(incumbent: Meta, metas: int, seed: int, r: int) -> list[Meta]:
    """The ways compared in round ``r``: the incumbent first, then variants of it. The same in every process."""
    rng = np.random.default_rng([seed, r])
    tried = [incumbent]
    for _ in range(200):
        if len(tried) == max(metas, 1):
            break
        copy = vary_meta(incumbent, rng)
        if copy not in tried:
            tried.append(copy)
    return tried


def level2_tag(r: int, index: int) -> str:
    return f"level2 round {r} meta {index}"


def round_record(r: int, tried: Sequence[Meta], runs: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], Meta]:
    """One round's entry and the way it chose: the largest gain in ``g`` per CPU-hour spent."""
    best = max(range(len(runs)), key=lambda i: runs[i]["gain_per_cpu_hour"])
    entry = {"round": r, "incumbent": tried[0].as_dict(), "chosen": tried[best].as_dict(), "changed": best != 0,
             "runs": [{"meta": run["meta"], "gain_per_cpu_hour": run["gain_per_cpu_hour"],
                       "g": run["progress"].get("g"), "verdict": run["progress"]["verdict"],
                       "cpu_seconds": run["cpu_seconds"], "measurements": run["measurements"],
                       "settings": run["searcher"]["settings"], "laws_known": len(run["searcher"]["laws"])}
                      for run in runs]}
    return entry, tried[best]


def run_level2(start: Searcher, *, rounds: int, metas: int, generations: int, seed: int, meta: Meta | None = None,
               measure: int = 20, workers: int = 1, client: Callable[[str], str] | None = None, first_round: int = 1,
               on_run: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Level 2: ways of running Level 1 compared by the gain in ``g`` they bought per CPU-hour.

    Each round the incumbent ``Meta`` runs again on new problems beside ``metas - 1`` variants of it; the best of the
    round is the next incumbent. Every run starts from the same searcher, so what is compared is the way of improving.

    ``first_round`` numbers the first round (a later round continued in another process draws other problems and
    other variants than the first); ``on_run`` is called with each Level 1 run as it ends, so a run cut short by a
    time limit leaves what it finished.
    """
    sealed = judge.seal()
    incumbent = meta or Meta()
    history = []
    for r in range(first_round, first_round + rounds):
        judge.check(sealed)
        tried = metas_for_round(incumbent, metas, seed, r)
        runs = []
        for i, m in enumerate(tried):
            runs.append(run_level1(start, generations=generations, meta=m, seed=seed, measure=measure, workers=workers,
                                   client=client, tag=level2_tag(r, i)))
            if on_run is not None:
                on_run({"round": r, "index": i, "meta": m.as_dict(), "gain_per_cpu_hour": runs[-1]["gain_per_cpu_hour"],
                        "measurements": runs[-1]["measurements"], "progress": runs[-1]["progress"],
                        "cpu_seconds": runs[-1]["cpu_seconds"], "settings": runs[-1]["searcher"]["settings"]})
        entry, incumbent = round_record(r, tried, runs)
        history.append(entry)
    judge.check(sealed)
    return {"rounds": history, "meta": incumbent.as_dict(), "sealed_files": len(sealed),
            "note": "the chosen way of improving won on the runs that chose it; a round in which the incumbent wins "
                    "again on new problems is the only confirmation here"}
