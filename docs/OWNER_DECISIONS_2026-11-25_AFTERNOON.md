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

## Correction to section 2, 25 November, 16:19 UTC (written before the 2-hour rung was read)

"The default search budget stays 90 seconds" is wrong as a description of the product, in the
agent's words to the owner and in this record. **The product has no default search budget of 90
seconds.** Read from the code (`apedp_dyn/ui/route_endpoint.py`, `apedp_dyn/ui/dashboard.html`):

- a table reading runs the first pass alone unless a budget is given (`search_seconds` in the
  request, or the variable `APEDP_UI_SEARCH_SECONDS`): the search stage is **off by default**;
- the dashboard's switch for the search stage asks for **30 seconds**;
- a request can ask for at most **300 seconds** (`MAX_SEARCH_SECONDS`), unless the process
  raises that ceiling for itself, as the measurements do.

Ninety seconds is the budget the development measurements of 24 and 25 November were made at (the
wider-conditions runs, and the first rung of the ladder). 240 seconds is the budget of the
preregistered benchmark runs.

*What the owner's "keep 90 seconds" therefore fixes:* nothing in the product's budgets changes:
the search stays off unless asked for, the dashboard asks for 30 seconds, the ceiling stays 300.
*The rule for the 2-hour rung* reads as before with one substitution: a change of those budgets is
proposed only if the condition of section 2 is met; otherwise they stay as they are and a longer
search stays an option on request. The condition, the rung and the tables are unchanged.

*What this leaves unmeasured:* the dashboard's own 30 seconds. No rung of the ladder ran at 30
seconds, so nothing recorded says what the product's switch finds against 90 seconds on the same
tables. A 30-second rung on the ladder's 40 tables would say, at about two hours of GitHub's
machines; it is a development measurement and is run after the 2-hour rung has been read.

## What the two rules of section 2 gave, 25 November, 16:55 UTC

Both runs ended earlier than expected and were read after the correction above was committed
(commit 51dcec8).

**The 2-hour rung** (`benchmarks/budget_ladder/dev_run1_cold_7200.json`). On the 20 tables it
shares with the other rungs (three and four inputs, noise 1 %): 4 laws reported at 90 seconds, 9
at 15 minutes, 12 at 2 hours; no report inaccurate at any rung; one law recovered by form, at 2
hours only. Two hours against 90 seconds, table by table: 8 gained, none lost, exact sign test
0.008. **The condition is met** (below 0.05, and no 2-hour report inaccurate). By the rule a
change of the product's search budgets is therefore *proposed*; it is the owner's to decide
(AGENTS.md 1.5) and nothing has been changed.

What the agent proposes, and what it does not:

- *Proposed:* a long search as something the product can be asked for and that does not hold a
  web request open: a reading started in the background with 15 minutes a search by default and 2
  hours at most, its result fetched when it is done. Most of the gain was there at 15 minutes (9
  of the 12; 2 hours against 15 minutes: 3 gained, none lost, sign test 0.25), at a seventh of the
  time.
- *Not proposed:* changing the dashboard's switch (30 seconds) or the ceiling of a web request
  (300 seconds). A request that waits two hours is not a web request, and the 30 seconds have not
  been measured.
- *What the measurement does not carry:* 20 tables, one noise level, three and four inputs, a
  development run with nothing frozen but this rule. Twelve reports with none wrong do not show
  the named limit on wrong reports held (that needs 29).

**Kept search processes** (`benchmarks/wider_conditions/dev_run5_kept_processes.json`, run
45645071039). (a) All 120 readings ended; all eight shards finished; no job was cut by its time
limit. (b) Three readings ended in an error, each "SVD did not converge in Linear Least Squares",
the fitting error the fresh-process runs also have; none names a search process or the pool. (c)
Against the fresh-process run on the same tables: 17 laws reported against 15; 4 tables gained, 2
lost, sign test 0.69; no inaccurate report. **Clean by all three conditions.** By the rule kept
processes are *proposed* as the default (`APEDP_SEARCH_PROCESSES=warm` as what the product does
unless told otherwise); the owner decides and nothing has been changed. Measured saving: 6.9
machine-hours against 8.0 for the 120 readings, 203 seconds a reading at the median against 231.
