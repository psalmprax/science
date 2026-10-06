---
id: c0a73111-3c3d-4b2d-8471-a19a31c494bd
title: Git decision from 29714401bef6
tags:
- decision
- git
created: 2026-10-01
updated: 2026-10-02
filenames:
- tests/test_coverage_audit.py
- tools/coverage_audit.py
links: []
kind: decision
status: proposed
superseded_by: null
deprecated_at: null
review_after: 2026-10-02
source_chat_id: null
created_at: 2026-10-01T13:41:39.276853734+00:00
summary: null
description: null
entities: []
related_files: []
related_entities: []
content_hash: 1244806ff9ab68286b60a67462b52b4c6a895d2534201e54cdfd1377e2e676f2
source_tool: buddy_memory_lifecycle:git
source_confidence: 0.8199999928474426
source_trajectory_id: null
source_message_range: null
source_commit: 29714401bef6683cb1efbc3a55eac5f49c8c4253
topic: null
last_used_at: null
use_count: 0
last_injected_at: null
dismissed_count: 0
source_content_hash: 1244806ff9ab68286b60a67462b52b4c6a895d2534201e54cdfd1377e2e676f2
review_needed: true
occurrences: 0
---

Git decision from 29714401bef6

Source commit: 29714401bef6
Paths: tests/test_coverage_audit.py, tools/coverage_audit.py
Summary: Make the coverage audit say why a module is unexercised, and commit it The audit that found the five dead modules was throwaway scripts in /tmp, and it got the answer wrong twice before it got it right, so nothing about it survived for the next person to reuse. This commits the working version, plus the two failure mod