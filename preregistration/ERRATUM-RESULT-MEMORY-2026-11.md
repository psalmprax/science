# Erratum to RESULT-MEMORY-2026-11 (25 November 2026)

The result stands as recorded: its two expectations held, on counts this erratum does not touch
(23 laws recovered with the memory beside the searches against 17 without; six gained, none lost;
no wrong law in 57 and 55 reports). The recorded files are not edited. One statement of the
result, made under "Reported without a criterion", is wrong, and this note corrects it.

**The statement.** "The memory's laws were the law itself in 26 of the 80 runs by the runner's
check: 15 without noise, 11 at 1 %."

**What is wrong with it.** The runner (`scripts/run_memory_laws.py`) took as "the memory's laws"
every candidate of the search stage whose text was not, character for character, one of the
searches' texts. The path rewrites every candidate when it refits its constants, so no text
matched, and every candidate of the stage was counted as one of the memory's: `handed_over` is 10
or 11 in all 80 rows, where the memory proposes three. The 26 is therefore the number of runs in
which **any** candidate of the stage, the searches' or the memory's, was the law.

**The corrected reading,** recounted on 25 November from the run's own rows
(`RESULT-MEMORY-2026-11/runs.jsonl.gz`; rows of the recount in
`benchmarks/ranked_memory/memory_result_recount.json`):

- the law was among the stage's candidates in 26 of the 80 runs (as recorded);
- in **17** of those the searches' own ten candidates already held it;
- in **9** it was there only with the memory beside them: 4 without noise, 5 at 1 %.

So the memory supplied a law the searches had not found in 9 runs of 80, not in 26. This agrees
with what the result's expectations measured (six more laws recovered, none lost) better than the
26 did.

**Who found it.** The independent reviewer of PREREG-RANKED-MEMORY-2026-11
(`docs/REVIEW_PREREG_RANKED_MEMORY_2026-11-25.md`, last item of finding 10), who rebuilt 72 of the
80 units and found the law among the memory's own three in 15 of them, 7 of which the searches
also held. The count above is over all 80 and asks a slightly different question (whether the
searches held the law, not which candidates were the memory's), which is why it is 9 and not 8.

**What else rested on the 26.** The sentences of the result that follow it ("Without noise the
route reported all 15", "At 1 % noise the route reported 6 of the 11") count runs in which the
stage held the law, not runs in which the memory supplied it, and should be read so. The
whiteboard's row for this result repeated the 26 and is corrected. The runner of
PREREG-RANKED-MEMORY-2026-11 does not make this comparison of texts.
