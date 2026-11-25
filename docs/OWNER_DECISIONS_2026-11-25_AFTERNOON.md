# Four decisions of the owner, 25 November 2026, afternoon

Asked "what would you have decided?", the agent proposed four things; the owner answered "yes to
all" (16:14 Berlin time). This file records each as it was put, what it commits to, and what it
does **not** cover. It was written at 15:15 UTC, **before the 2-hour rung of the budget ladder
was read** (that rung was still running; its record is collected at about 17:50 UTC).

Nothing in this file changes what is in force. Where a decision leads to a proposal for force, the
owner decides again when the measurement is in (AGENTS.md 1.5).

## 1. The learned order is tested through the referee, beside the memory in force

*As put:* "test it through the referee, added beside the memory in force rather than replacing
it. Beside means the memory still hands over its own three candidates and the ranked proposer
adds three more. [...] a frozen, run-once test on fresh problems through the full table route.
Pass mark: more laws reported, with wrong reports inside the 5 % limit you set today. The repo's
rules require an independent reviewer before I freeze the plan, so this needs your go for a
separate reviewer."

*Commits to:* one preregistered comparison, run once, of the table route in force with and
without the ranked proposer's candidates, on laws drawn after the proposer and its order are
fixed; a separate reviewer before freezing and before recording (AGENTS.md 1.6). The owner's yes
is the go for that reviewer.

*Does not cover:* putting the ranked proposer in force. That is a second decision, after the
result.

## 2. The default search budget stays 90 seconds; the rule for changing it is fixed now

*As put:* "keep 90 seconds. [...] I would fix the rule now, before the 2-hour rung is read: the
default changes only if the longer rung reports clearly more on its tables with no wrong report.
Otherwise the long search stays an option on request. Kept processes save about a fifth of the
time, but they are new process-handling code in the place where the hangs were. I would want one
more clean full run before proposing them as default."

*The rule, in terms that can be checked:*

- **Fifteen minutes** (already read when this was written: 18 laws reported against 12 at 90
  seconds on the same 40 tables, 9 tables gained and 3 lost, sign test 0.15, none inaccurate):
  not clearly more. The default does not change on it.
- **Two hours** (not read when this was written): the default budget is proposed for change only
  if, on the 20 tables that rung shares with the 90-second rung, the exact sign test of tables
  gained against tables lost in laws reported is below 0.05 **and** none of the 2-hour rung's
  reports is inaccurate. Anything else: the default stays 90 seconds, and a longer search stays
  what it is now, an option on request (`search_seconds`, `APEDP_UI_MAX_SEARCH_SECONDS`).
- **Kept search processes** are not proposed as the default until one more full run of the 120
  wider-conditions tables with kept processes ends clean. *Clean* means all three of: (a) every
  one of the 120 readings ends, none stopped and no job cut by its time limit; (b) at most three
  readings end in an error, and none of those errors names a search process or the pool (the
  fresh-process runs had one to three readings a run ending in a fitting error); (c) against the
  fresh-process run on the same tables (`benchmarks/wider_conditions/dev_run4.json`), tables
  gained and lost in laws reported do not differ beyond chance (sign test not below 0.05) and no
  more than two reports are inaccurate (the most any fresh-process run had). If it is clean, the
  agent proposes kept processes as the default and the owner decides.

## 3. The order of the ideas from outside

*As put:* "split wide tables first. It answers a measured zero, needs only CPU, and extra search
time did not move that zero. Second is letting failed searches teach the order, since the order
and its measurement now exist. Third is laws about laws. I would not spend GPU money yet."

*Commits to:* development studies in that order (`docs/IDEAS_FROM_OUTSIDE_2026-11-25.md`: C1,
A1, C4), each with its pass mark, each a development study before anything is proposed. No paid
GPU is used.

## 4. The first experiment on GF(2^12) multiplication aims at 41 products or fewer

*As put:* "approve the target of 41 products or fewer, and run it without Colab. I would first
run the existing search on GitHub's machines with no model, as the baseline any coding model has
to beat. A model and a call budget come only after that baseline exists."

*Commits to:* the target of `docs/GF2_12_MULTIPLICATION_RECORD_2026-11-25.md`, proposal 1: a
decomposition with 41 or fewer products, starting from the verified 42; and a recorded baseline of
what the existing search reaches with no language model, on GitHub's machines.

*Does not cover:* proposals 2 and 3 of that document (correcting section 4.1 of
`specifications/APEDP_QUANTUM_DISCOVERY_V1.md`, and measuring the 42 in the decoder benchmark).
They stay open. No specification file is edited on the strength of this decision.
