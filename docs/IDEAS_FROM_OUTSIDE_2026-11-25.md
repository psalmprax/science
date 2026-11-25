# Ideas from outside, set against what this system measures (2026-11-25)

Asked for by the owner: what is not yet part of this system that could make it much stronger,
from papers, textbooks and talks. This is a reading list turned into proposals. **Nothing here
is in force, nothing here is a result, and none of it is a recipe for superintelligence**: the
strongest self-improving systems in print say themselves what they have not shown (section 5).
Each idea has a test with a pass mark, so that it can be dropped when it fails.

## 1. What was read, and what was not

* **Papers** (arXiv and journals), found by search and read in abstract or in full: listed with
  each idea below.
* **Talks and interviews, by transcript**: the Machine Learning Street Talk interview with the
  authors of AlphaEvolve (14 May 2025; a transcript page), and Dwarkesh Patel's interview with
  Terence Tao (transcript at dwarkesh.com/p/terence-tao).
* **Talks, by written notes only**: Sutton's keynote on the OaK architecture (RLC 2025), Miles
  Cranmer's colloquium "The Era of Scientific Foundation Models" (26 March 2026, abstract), the
  ARC Prize 2025 technical report, and the README of Karpathy's `autoresearch` (March 2026).
* **Not done**: two years of YouTube were not mined. There is no way from this workspace to pull
  video transcripts in bulk, and the site was not reachable when tried. What stands above is two
  interview transcripts and notes on four talks, not a sweep.

## 2. Where the system is weak, by its own measurements

| What was measured | Where |
|---|---|
| The law is on the memory's list but not among the first three handed to the referee | `benchmarks/order_score/` (this week) |
| At higher levels of the curriculum the law is not on the list at all | the same |
| The survivors cannot be told apart on the table (`equivalent_class`) in about half of the readings | wider-conditions runs 3 and 4 |
| No law recovered by form in 240 readings of tables with three, four and five inputs | the same |
| Each search is one core for 90 seconds in a process started for that table | `budget-ladder` (running) |
| The recursion chose between searchers on ten problems, with a rule that rewards cheapness on ties | `benchmarks/recursion/dev_search_run2.json` |

An idea is worth its cost here only if it moves one of these rows.

## 3. What the system already has (so it is not proposed again)

A referee that proposers cannot overrule; a memory of confirmed laws (their one-input parts and
whole shapes); a library of building blocks learned by compression (`recursion/library.py`); an
e-graph that shows two expressions equal (`recursion/egraph.py`) and an SMT check on a box
(`recursion/smt.py`); a curriculum that moves its own level; three levels of improvement
(`recursion/levels.py`); a language model over equations used as a prior (`apedp_mmlm`); hosted
language models as proposers and as writers of candidate generators (`evolution/`); a contrast
stage that measures where candidates disagree when a pool of further cases exists; a units
reduction where units are given; and, since this week, the order score, a learned order, a store
of solved cases found by what a table looks like, kept search processes and a ladder of budgets.

## 4. The ideas

Ordered by how directly they answer a row of section 2. "Pass mark" always means: on the 300
fixed problems of the order score, a paired sign test below 0.05 **and** a gain at least as large
as the smallest one the score can show at 80 % power (`benchmarks/order_score/dev_run1.json`,
`what_can_be_seen`), unless another mark is named.

### A. Make the proposer learn from everything it does

**A1. Every failed search is a solved problem for some other table (hindsight relabelling).**
Source: SOAR (arXiv:2507.14172), CodeIt (arXiv:2402.04858). A search that misses the law of table
T still fitted hundreds of candidates; each candidate is the exact law of the table it produces
itself. So every search yields correct (table, law) pairs for free, in the very distribution the
search explores. *Here:* only confirmed laws are kept; a miss teaches nothing. *What it adds:*
unlimited training pairs for the learned order (A3) and for a trained proposer (A2), with no
human and no benchmark. *Test:* the learned order trained on relabelled pairs plus curriculum
pairs against curriculum pairs alone, at equal problems. *Cost:* CPU only. *Risk:* pairs made
from over-fitted candidates teach over-fitted shapes; keep only candidates no longer than a
bound in bits.

**A2. A proposer that sees the table (a small trained network).** Source: DreamCoder's
recognition model (arXiv:2006.08381), neural symbolic regression, a fine-tuned language-model
regressor (arXiv:2508.09897), DIGEST (arXiv:2610.33880, October 2026: a language model tuned
under the guidance of what the search has discovered). *Here:* the equation language model is a prior that never sees the
table; nothing trained proposes from the rows. *What it adds:* laws on the list that the memory's
parts cannot compose. *Test:* order score, "found on the list" by level. *Cost:* a GPU session
(no Colab plugin or connector exists here; Hugging Face Jobs on the owner's account, or the
notebook route). **Not built.**

**A3. An order of candidates that is learned.** Textbook: learning to rank, and the
one-standard-error rule of model selection. **Built this week** (`proposers/learned_order.py`),
measured by `scripts/study_learned_order.py`.

**A4. Solved cases found again by what a table looks like.** Source: LaSR's concept library
(arXiv:2409.09359), retrieval-augmented symbolic regression (SR-LLM, PNAS 2025). **Built this
week** (`proposers/experience.py`), under measurement with a language model
(`scripts/study_llm_retrieval.py`).

**A5. Refine, do not propose once (refinement loops and test-time training).** Source: the ARC
Prize 2025 report names the refinement loop as the year's theme (arXiv:2601.10904); a 7M-weight
network that improves its own answer over up to 16 steps (arXiv:2510.04871). *Here:* a proposer
hands over a list and stops. *What it adds:* (i) propose a correction to the best candidate's
residual, several rounds; (ii) before proposing, train the order on generated tables that look
like this one (same inputs, same signature region). *Test:* order score at equal fits, one round
against three. *Cost:* CPU.

**A6. Tasks the system sets itself at the edge of what it solves.** Source: Absolute Zero
(arXiv:2505.03335), POET (arXiv:1901.01753), learning-progress curricula. *Here:* the curriculum
moves one level up or down by fixed thresholds. *What it adds:* draw problems where the solve
rate is changing fastest, per kind of block, not per level. *Test:* order score on the fixed 300
after equal CPU, against the curriculum as it is. *Cost:* CPU.

### B. Spend the search's time where it pays

**B1. A fixed budget, one number, one file.** Source: Karpathy's `autoresearch` (March 2026): an
agent edits one file, every experiment runs for the same wall-clock time, one metric decides
keep or discard. *Here:* `levels.better` lets a copy win by being cheaper at equal solves, and it
halved the search time instead of solving more. *What it adds:* the recursion compares copies at
equal CPU on the order score's mean reciprocal place. *Test:* generations 1 to 4 under the new
rule: the score must not fall and the search time must not shrink. *Cost:* CPU; a change of
what the recursion optimises, so the owner's word.

**B2. An archive, not one incumbent.** Source: Darwin Gödel Machine (arXiv:2505.22954), MAP-Elites
(arXiv:1504.04909), the AlphaEvolve authors on keeping weaker candidates as stepping stones.
*Here:* the candidate-generator loop keeps an archive (`evolution/operator.py`); the recursion
keeps one searcher. *Test:*
best order score after four generations, archive against incumbent, three seeds.

**B3. Racing instead of equal time.** Source: successive halving and Hyperband
(arXiv:1603.06560); the evaluation cascade of AlphaEvolve (cheap check for many, dear check for
few). *Here:* every table gets the same seconds and every member the same share. *What it adds:*
short searches for all tables, long ones only where the best candidate is near the line. *Test:*
laws reported per CPU-hour on the ladder's tables, with wrong reports no higher. *Cost:* CPU.

**B4. A bandit over members and models; refuse what was already tried.** Source: ShinkaEvolve
(arXiv:2509.19349): islands, novelty rejection, bandit choice of model. *Here:* four members,
fixed. The e-graph could refuse a candidate equal to one already fitted (EGG-SR,
arXiv:2511.05849, puts e-graphs inside the search for this). *Test:* distinct structures fitted
per CPU-second; order score.

**B5. Search the program that builds the answer.** Source: the AlphaEvolve interview: evolve the
search routine or the constructor, not the object; build the time limit into the problem; mix a
fast model for volume with a strong one for quality. *Here:* this is step 4. For GF(2^12) the
object to evolve is the search routine around the flip graph (arXiv:2212.01175; polynomial
multiplication over Z_2 up to degree 10 in arXiv:2502.06264; nothing found on extension fields),
seeded with the verified 42 products. Waits for the owner's word on the target.

### C. Cure the tables that cannot be decided

**C1. Split a wide table before searching it.** Textbook: AI Feynman (arXiv:1905.11481,
arXiv:2006.10782) fits a smooth model and tests it for additive and multiplicative separability
and symmetries; Cranmer et al. read laws off the parts of a trained network (arXiv:2006.11287).
*Here:* nothing in the route in force splits a table; four and five inputs are searched whole.
*Test:* the wider-conditions tables with four and five inputs: laws recovered by form (now 0 of
160), wrong reports no higher. *Cost:* CPU. **The most direct answer to a measured failure.**

**C2. Say which rows would decide it.** Source: Bayesian experimental design for symbolic
discovery (arXiv:2211.15860), query by committee (arXiv:2305.10379), a pool of model families
weighted by evidence with the next experiment chosen by expected information
(arXiv:2610.25747, simulated benchmarks only). *Here:* the contrast stage measures where
candidates disagree when a pool exists; the table reading ends at `equivalent_class`. *What it
adds:* the reading returns the input ranges where the surviving laws differ by more than the
noise, as a request. *Test:* on curriculum tables that end undecided, rows needed to decide,
chosen against random: half as many.

**C3. A new law must agree with the confirmed ones where they were confirmed.** Source:
Cranmer's 2026 colloquium: physics extrapolates because a new model has to contain the old
ones. *Here:* the memory hands over parts; it does not constrain. *Test:* curriculum families
built as extensions of earlier laws; order score with the constraint as a feature of the order.

**C4. Laws about laws.** Kepler's third law is a law of the constants of six orbits (Tao, in
the interview, on how little data it rested on). *Here:* the memory drops every fitted constant.
*What it adds:* when one shape is confirmed on several tables, its constants become a new table
over the conditions of those tables, and the route reads that table like any other. *Test:* a
curriculum of families (one shape, constants that follow a hidden law of a condition): share of
hidden laws reported. **Not found in the system; no new search is needed, only bookkeeping.**

**C5. Derive, then test.** Source: AI-Descartes (Nature Communications 2023) and AI-Hilbert
(Nature Communications 2024) join data with what follows from known theory. *Here:* two
confirmed laws that share a variable imply a third (eliminate the variable); the e-graph and SMT
tools exist. A derived law that survives the referee on rows nobody fitted is knowledge the
search never had to find. *Test:* pairs of curriculum laws with a shared variable: derived laws
confirmed on fresh rows.

### D. Keep the referee sound as the volume grows

**D1. Tests that may be repeated (e-values).** Source: POPPER (arXiv:2502.09858): falsification
tests run one after another, combined as e-values, with the false-positive rate held; textbook:
anytime-valid inference (arXiv:2210.01948). *Here:* nothing in the referee is anytime-valid; the
recursion and the ladder look at the same problems many times. *Test:* on tables of pure noise
under optional stopping, reports stay under the level. A change to what is in force: the
owner's word, a pinned threshold, a CHANGELOG entry.

**D2. Breadth, at a low rate, with the checking counted.** Tao's sweeps: one or two in a hundred
hard problems fall, and that is worth having at scale; checking, not proposing, is what runs
short. *Here:* this is the case for the referee being the product. Plan budgets by the number of
tables attempted, not by the hope for any one.

### E. The meta level

**E1. Let the improver edit how it remembers.** Source: ALMA (arXiv:2602.07755): a meta agent
searches over memory designs written as code (what is stored, how it is found, how it is
updated). *Here:* three hand-made memories (motifs, library, cases). *Test:* order score of the
best design found against the three. Evidence so far is from game-like benchmarks only.

**E2. Let the improver edit itself.** Source: HyperAgents (arXiv:2603.19461). *Here:* level 2
of the recursion changes how level 1 searches for settings. See section 5 before spending on it.

## 5. What the sources say not to expect

* The Darwin Gödel Machine replaces proof by measurement on a benchmark; its authors ran it
  sandboxed with human oversight. HyperAgents reports gains at the meta level that carry across
  domains and does not show improvement without bound.
* A 2026 survey of self-improving agents (arXiv:2607.13104) lists what a working loop needs
  (an evaluator the loop cannot touch, outcomes that can be verified, original data kept,
  diversity, versions and rollback) and how loops fail (collapse, an evaluator that drifts
  toward the proposer, reward hacking). This repository's rules already hold the first list.
* AlphaEvolve works where a score can be computed and climbed; its authors say a yes-or-no
  reward is hard to climb and that sometimes nothing better is found.
* A language model that proposes laws can be reading them back: LLM-SRBench (arXiv:2504.10415)
  was built because of that, and a 2026 study finds that a fixed library of operators with no
  language model solves most of that benchmark's synthetic tasks (arXiv:2607.28684). A gain from
  a language model has to be shown against that kind of control, on laws invented here.

## 6. Proposed order

1. C1 (split wide tables): answers a measured zero, CPU only.
2. A1 with A3 (hindsight pairs for the learned order): the loop that learns from its own work.
3. C4 (laws about laws): bookkeeping, and the first thing here that compounds.
4. B1 and B2 (what the recursion optimises): needs the owner's word.
5. A2 (a trained proposer): needs a GPU session.
6. D1 (repeatable tests): needs the owner's word and a pinned threshold.

Each is a development study first, with its pass mark, before anything is proposed for force.
