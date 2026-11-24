"""A real language model as the proposer of moves in the GF(2^12) decomposition search (MMLM-POC-001).

Status: experimental

``scripts/run_mmlm_poc_001.py`` has an arm that was labelled "Frontier LLM Agent" and is hand-written
algebra: it picks among four fixed strategies and adds made-up token counts (found by an outside
audit, 2026-11-24). This module is the arm that label promised. A language model is shown the
current decomposition and the exact checker's failure report and replies with **one move**; the
checker applies it and says what it did. The model never decides whether its move is kept: a move
is kept if the exact defect count falls, as in the other arms.

The model is any ``client(prompt) -> text`` callable (``apedp_dyn.proposers.llm_https.nvidia_client``
for a hosted model; a scripted stand-in in the tests). This module has no network code and never
sees a key. Tokens are the service's own counts (``client.last["usage"]``), or ``None`` where the
client gives none; nothing is estimated.

Moves, the same families the hand-written arm uses, so that the comparison is of who chooses:

* ``{"move": "set_term", "term": r, "u": [bits], "v": [bits]}``: replace product ``r`` by
  ``(sum of a_i for i in u) * (sum of b_j for j in v)``;
* ``{"move": "xor_terms", "into": r, "from": s, "bits": [bits]}``: add those bits of product ``s``'s
  two factors into product ``r``'s;
* ``{"move": "frobenius", "term": r}``: square both factors of product ``r`` in the field.

A reply that is not one such move is recorded as invalid and costs its call.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from apedp_dyn.quantum.field_tensor_search import (
    DEGREE,
    MODULUS_INT,
    diagnose_gf2_bilinear_system,
    gf2_poly_mod,
)

MOVES = ("set_term", "xor_terms", "frobenius")
FAILING_SHOWN = 30           # failing coordinates listed in a prompt
HISTORY_SHOWN = 8            # earlier moves listed in a prompt, with what each did
REPLY_KEPT = 1500            # characters of a reply kept in the log
_OBJECT = re.compile(r"\{[^{}]*\}")


def bits_of(vector: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(vector)]


def frobenius(vector: np.ndarray) -> np.ndarray:
    """The field's Frobenius map (squaring) on an element given by its bits."""
    value = 0
    for i in range(DEGREE):
        if vector[i]:
            value ^= 1 << i
    squared = gf2_poly_mod(value * value, MODULUS_INT)
    out: np.ndarray = np.array([(squared >> i) & 1 for i in range(DEGREE)], dtype=np.uint8)
    return out


def describe(U: np.ndarray, V: np.ndarray, diagnostics: Any, history: list[dict[str, Any]]) -> str:
    """The prompt: the task, the current products, the exact checker's report, and what earlier moves did."""
    rank = len(U)
    lines = [
        "You are helping to search for a short formula for multiplying two elements of the field GF(2^12),",
        "with modulus x^12 + x^6 + x^4 + x + 1. An element is 12 bits a_0..a_11 (and b_0..b_11).",
        f"A decomposition is a list of {rank} products. Product r is (XOR of a_i for i in u_r) AND (XOR of b_j for j in v_r).",
        "It is correct if each of the 12 output bits is an XOR of some of the products. An exact checker decides that.",
        "",
        "Current products (r: u | v), bits numbered 0 to 11:",
    ]
    lines += [f"{r}: {bits_of(U[r])} | {bits_of(V[r])}" for r in range(rank)]
    failing = [tuple(int(v) for v in c) for c in diagnostics.failing_coordinates[:FAILING_SHOWN]]
    lines += [
        "",
        "The checker's report on these products:",
        f"- wrong entries (i, j, k) of the multiplication table: {diagnostics.defect_count}",
        f"- independent products: {diagnostics.rank_A} of {rank}; products that add nothing: {list(diagnostics.redundant_products)}",
        f"- directions of the table the products cannot reach: {diagnostics.rank_deficiency}",
        f"- some wrong entries (a_i * b_j contributing to output bit k): {failing}",
    ]
    if history:
        lines += ["", "Your earlier moves and what the checker found:"]
        lines += [f"- {h['move']} -> wrong entries {h['before']} to {h['after']}, {h['outcome']}" for h in history[-HISTORY_SHOWN:]]
    lines += [
        "",
        "Propose ONE move that should lower the number of wrong entries. A product that adds nothing is free to replace.",
        "Reply with exactly one line of JSON and nothing else, one of:",
        '{"move": "set_term", "term": r, "u": [bits], "v": [bits]}',
        '{"move": "xor_terms", "into": r, "from": s, "bits": [bits]}',
        '{"move": "frobenius", "term": r}',
    ]
    return "\n".join(lines)


def _bits(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not all(isinstance(b, int) and not isinstance(b, bool) and 0 <= b < DEGREE for b in value):
        return None
    return sorted(set(value))


def _term(got: dict[str, Any], key: str, rank: int) -> int | None:
    value = got.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < rank else None


def parse_move(reply: str, rank: int) -> dict[str, Any] | None:
    """The move in a reply, or ``None`` if the reply holds no valid one. The last JSON object in the text is read."""
    for text in reversed(_OBJECT.findall(str(reply))):
        try:
            got = json.loads(text)
        except ValueError:
            continue
        if not isinstance(got, dict) or got.get("move") not in MOVES:
            continue
        if got["move"] == "set_term":
            r, u, v = _term(got, "term", rank), _bits(got.get("u")), _bits(got.get("v"))
            if r is not None and u and v:
                return {"move": "set_term", "term": r, "u": u, "v": v}
        elif got["move"] == "xor_terms":
            r, s, bits = _term(got, "into", rank), _term(got, "from", rank), _bits(got.get("bits"))
            if r is not None and s is not None and r != s and bits:
                return {"move": "xor_terms", "into": r, "from": s, "bits": bits}
        else:
            r = _term(got, "term", rank)
            if r is not None:
                return {"move": "frobenius", "term": r}
    return None


def apply_move(U: np.ndarray, V: np.ndarray, move: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The products after a move. The inputs are not changed."""
    U, V = U.copy(), V.copy()
    if move["move"] == "set_term":
        U[move["term"]] = 0
        V[move["term"]] = 0
        U[move["term"], move["u"]] = 1
        V[move["term"], move["v"]] = 1
    elif move["move"] == "xor_terms":
        U[move["into"], move["bits"]] ^= U[move["from"], move["bits"]]
        V[move["into"], move["bits"]] ^= V[move["from"], move["bits"]]
    else:
        U[move["term"]] = frobenius(U[move["term"]])
        V[move["term"]] = frobenius(V[move["term"]])
    return U, V


def run_llm_arm(T: np.ndarray, client: Callable[[str], str], start_U: np.ndarray, start_V: np.ndarray, *,
                max_steps: int = 60) -> dict[str, Any]:
    """The search with a language model choosing each move, from the given products, for ``max_steps`` calls.

    Every call, reply, move and outcome is in ``log``. A call the service refuses ends the arm there, with the
    reason; what was reached stands.
    """
    U, V = start_U.copy(), start_V.copy()
    diagnostics, weights = diagnose_gf2_bilinear_system(U, V, T)
    start_defect = best = int(diagnostics.defect_count)
    history: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    tokens: dict[str, int] = {}
    usage_missing = invalid = kept = 0
    stopped = None
    solution = (U, V, weights) if diagnostics.is_consistent else None
    for step in range(1, max_steps + 1):
        if solution is not None:
            break
        prompt = describe(U, V, diagnostics, history)
        try:
            reply = client(prompt)
        except Exception as exc:  # noqa: BLE001 - whatever stops the service ends the arm with its reason; the log stands
            stopped = f"{type(exc).__name__}: {exc}"[:200]
            break
        usage = dict((getattr(client, "last", None) or {}).get("usage") or {})
        if usage:
            for key, value in usage.items():
                tokens[key] = tokens.get(key, 0) + int(value)
        else:
            usage_missing += 1
        entry: dict[str, Any] = {"step": step, "prompt_characters": len(prompt), "reply": str(reply)[:REPLY_KEPT], "usage": usage,
                                 "before": best}
        move = parse_move(reply, len(U))
        if move is None:
            invalid += 1
            entry.update({"move": None, "after": best, "outcome": "invalid reply"})
            history.append({"move": "(not a valid move)", "before": best, "after": best, "outcome": "invalid reply"})
            log.append(entry)
            continue
        cand_U, cand_V = apply_move(U, V, move)
        cand_diagnostics, cand_weights = diagnose_gf2_bilinear_system(cand_U, cand_V, T)
        after = int(cand_diagnostics.defect_count)
        improved = cand_diagnostics.is_consistent or after < best
        entry.update({"move": move, "after": after, "outcome": "kept" if improved else "rejected"})
        history.append({"move": json.dumps(move), "before": best, "after": after, "outcome": entry["outcome"]})
        log.append(entry)
        if improved:
            kept += 1
            U, V, diagnostics, best = cand_U, cand_V, cand_diagnostics, after
            if cand_diagnostics.is_consistent:
                solution = (cand_U, cand_V, cand_weights)
    return {
        "arm": f"language model ({getattr(client, 'model', 'unnamed')})", "target_rank": len(start_U),
        "start_defect": start_defect, "best_defect": 0 if solution is not None else best, "solved": solution is not None,
        "calls": len(log), "moves_kept": kept, "invalid_replies": invalid, "tokens": tokens or None,
        "calls_without_a_token_count": usage_missing, "stopped": stopped, "log": log,
        "products": {"U": U.astype(int).tolist(), "V": V.astype(int).tolist()},
    }
