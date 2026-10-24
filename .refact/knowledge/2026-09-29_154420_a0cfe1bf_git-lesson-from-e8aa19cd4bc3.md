---
id: b8c80320-c060-4370-8e5b-c6881734599c
title: Git lesson from e8aa19cd4bc3
tags:
- git
- lesson
created: 2026-09-29
updated: 2026-10-06
filenames:
- apedp_dyn/model/equations.py
- apedp_dyn/problems/base.py
- apedp_dyn/simulation/ode_solver.py
- tests/test_dyn_problem_zoo.py
links: []
kind: lesson
status: proposed
superseded_by: null
deprecated_at: null
review_after: 2026-10-06
source_chat_id: null
created_at: 2026-09-29T13:44:20.547975642+00:00
summary: null
description: null
entities: []
related_files: []
related_entities: []
content_hash: 2311bc17e4d2bfeffebff9826e2347eb72e1a8fcfa0f99751b9a70c7dd0e8feb
source_tool: buddy_memory_lifecycle:git
source_confidence: 0.8600000143051147
source_trajectory_id: null
source_message_range: null
source_commit: e8aa19cd4bc32d020992026e9d625d4900c143e4
topic: null
last_used_at: null
use_count: 0
last_injected_at: null
dismissed_count: 0
source_content_hash: 2311bc17e4d2bfeffebff9826e2347eb72e1a8fcfa0f99751b9a70c7dd0e8feb
review_needed: true
occurrences: 2
last_observed: 2026-09-29T13:44:21.446064685+00:00
---

Git lesson from e8aa19cd4bc3

Source commit: e8aa19cd4bc3
Paths: apedp_dyn/model/equations.py, apedp_dyn/problems/base.py, apedp_dyn/simulation/ode_solver.py, tests/test_dyn_problem_zoo.py
Summary: fix: SymPy was hijacking physics symbol names; add a zoo smoke test Seven of the eleven new problem modules could not be loaded at all, and nothing in the suite noticed, because no test loaded the problems. Two independent causes, both in parsing rather than physics: SymPy pre-binds short names. `I` is the imaginary un