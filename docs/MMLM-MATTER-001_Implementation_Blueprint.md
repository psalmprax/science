# MMLM-MATTER-001 — Implementation Blueprint

**Status:** Implementation plan for post-20 January 2027 work  
**Scope:** One process (FDM printing in PLA), one mechanism family initially (planar one-degree-of-freedom lower-pair linkages), software-first validation  
**Primary research question:** Under matched downstream search budgets and identical checkers, does LLM-guided topology selection find better verified mechanisms than the strongest frozen conventional topology-selection strategy?  
**Primary comparison:** P1 (LLM-guided topology selection) versus P0 (classical topology selection)  
**Ablation:** P2 (LLM without numerical optimiser)  
**Referee:** APEDP-X, independent of all proposers

---

## 0. Executive decision and non-goals

Build a bounded, reproducible software experiment. Do not build a universal Matter Compiler, a general manufacturing platform, or a physical manufacturing system.

The transferable idea is:

> Convert a design into a typed sequence of operations, check every operation, measure the result in the software environment, identify deviations, and correct the design or plan. Every claim must state exactly what kind of evidence supports it.

The project does **not** claim that a design was physically manufactured or that a simulator proves real-world reliability. Physical construction is outside the core scope. A future optional printer phase may produce L5 evidence, but no L5 claim is permitted before actual construction and measurement.

### 0.1 Fixed initial scope

- Manufacturing process: fused-filament fabrication (FDM/FFF).
- Material model: PLA only.
- Mechanisms: planar, one-degree-of-freedom, revolute-joint lower-pair linkages.
- Link-count catalogue: 4, 6, 8 and 10 links, counting the ground link.
- Primary design problem: path generation and topology selection.
- Secondary problems: function generation and print-in-place feasibility.
- Initial software stack: Python orchestration; a CAD/geometry kernel; an existing slicer; independent kinematics implementations; an FEA solver only after its benchmark gate passes.
- No proprietary Atomic Machines process is assumed or reproduced.

### 0.2 Explicit non-goals

- Universal process-description language.
- Universal physical-object compiler.
- Claiming that software-only results establish that an object works in reality.
- Building a printer, metrology laboratory, or custom toolpath generator.
- Claiming a four-bar is impossible merely because a search failed to find one.
- Claiming LLM advantage from a single attractive example.
- Adding a new permanent MMLM division before Phase 1 demonstrates a measurable contribution.

### 0.3 Decision rule

The project continues beyond Phase 1 only if P1 demonstrates a preregistered, statistically supported advantage over P0 on the topology-selection benchmark (T3), at a realistic fixed search budget, using the same optimisation and verification tools. If it does not, record the negative result and stop the LLM-specific Matter work. Conventional CAD and mechanism engineering may continue independently if useful.

---

## 1. Research hypothesis and claims

### 1.1 Primary hypothesis

**H1:** On a fixed hidden set of 200 mechanism-synthesis tasks, P1 finds more valid designs, or lower-cost valid designs, than P0 under the same downstream topology-evaluation and optimiser budget.

P1's permitted contribution is selecting and ordering which mechanism topologies/inversions should be investigated. It may use the common optimiser and common checkers. It may not use a stronger optimiser, additional objective evaluations, privileged task information, or different acceptance criteria.

### 1.2 Secondary hypotheses

- **H2:** The P1 advantage, if present, is largest at intermediate budgets and decreases as the common topology catalogue approaches exhaustive evaluation.
- **H3:** P1 may contribute more on T3 (topology selection) and T4 (printability constraints) than on ordinary four-bar T1/T2 tasks, where established synthesis methods are strong.
- **H4:** P2 (LLM without the optimiser) will provide an ablation showing whether direct LLM design is useful without deterministic numerical search.
- **H5:** A closed-loop corrector in simulation can outperform a no-correction baseline under a frozen, data-supported perturbation model. This establishes only model-conditional simulation performance, not real printer reliability.

### 1.3 Claims that are explicitly not being tested

- That an LLM has discovered a new general theory of mechanisms.
- That a failed search proves no mechanism exists.
- That published PLA dimensional measurements establish the variation of printed linkage joints unless repeated and relevant measurements are actually present.
- That simulated robustness is the same as physical reliability.

---

## 2. Benchmark structure

### 2.1 Task families

| Task | Purpose | Main checker | Status in the experiment |
|---|---|---|---|
| T1 — Path generation | Trace a supplied ordered planar path | Kinematics plus ordered path metric | Four-bar foundation and secondary benchmark |
| T2 — Function generation | Approximate a specified input/output angle relationship | Freudenstein-equation residual and kinematics | Secondary benchmark |
| T3 — Topology selection | Select a linkage topology/inversion and dimensions for a target motion/path | Topology validity, kinematics and path metric | **Primary P0-versus-P1 experiment** |
| T4 — Print-in-place | Convert an accepted T1/T3 design into printable parts and captured joints | Geometry, rules, collision and slicer | Phase 2 engineering benchmark |

T1 and T2 validate the synthesis and checker foundation. T3 is the scientific test of topology-selection intelligence. T4 tests whether an accepted mechanism can be translated into a constrained printable design.

### 2.2 T3 mechanism universe

The initial universe is one-degree-of-freedom planar kinematic chains made only from revolute lower pairs. Count the ground link as a link. Do not silently include gears, cams, higher pairs, prismatic joints, compliant mechanisms, or spatial linkages.

The reference enumeration reports the following numbers of distinct base chains for one degree of freedom:

| Links | Base-chain count |
|---:|---:|
| 4 | 1 |
| 6 | 2 |
| 8 | 16 |
| 10 | 230 |

These are **base-chain counts under the cited enumeration conventions**, not the number of usable mechanisms after inversions. The cited table excludes unique kinematic inversions and certain joint classes. The implementation must expand ground/input/output choices and remove equivalent configurations by graph canonicalisation. Do not report 249 as the total number of mechanisms.

### 2.3 Topology expansion

For every base chain:

1. Build a canonical labelled graph representation.
2. Enumerate possible ground-link choices.
3. Enumerate permitted input-link or input-joint choices.
4. Enumerate permitted output-link, output-joint, or coupler-point choices appropriate to the task.
5. Apply mobility and connectivity checks.
6. Remove graph-isomorphic duplicates while preserving physically distinct inversions.
7. Store the mapping from each expanded mechanism to its base-chain parent.
8. Store the canonical graph hash and the exact enumeration convention/version.

The exact number of expanded topologies is an empirical Phase 0 output. Do not estimate it in advance by multiplying the base-chain counts by a generic factor.

### 2.4 Mobility and admissibility

For the initial planar all-revolute lower-pair universe, use the appropriate planar mobility relation as a screening check, then validate the actual graph and constraints. Do not rely on a single scalar mobility formula to detect every overconstrained or special-case mechanism.

Reject or separately tag:

- Disconnected graphs.
- Mechanisms with mobility other than one under the declared model.
- Duplicate links/joints caused by invalid graph construction.
- Degenerate links or coincident pivots not permitted by the design rules.
- Mechanisms with no feasible assembly configuration for the task.

### 2.5 T3 task set: 200 hidden tasks

Generate 200 hidden tasks with reproducible seeds, balanced as evenly as possible across planted 6-, 8- and 10-link source mechanisms (for example, 66/67/67). Freeze the final allocation before the hidden evaluation begins.

Each task should contain:

- Task ID and hidden seed.
- Required input motion or input/output relationship.
- Ordered target path or function samples.
- Coordinate frame and units.
- Allowed geometric envelope.
- Required accuracy/tolerance.
- Permitted input range and direction.
- Load requirement, if any.
- Any manufacturing constraints relevant to T4.
- Ground-truth planted topology and dimensions, sealed from proposers.
- A reference implementation that can reproduce the planted motion.

The planted source mechanism demonstrates that at least one construction exists. It does **not** prove that the planted topology is minimal. A proposer is allowed to find a simpler 4-, 6- or 8-link mechanism that passes the same task tolerance; that is a legitimate improvement and should win on the cost ordering.

If a task is described as “no four-bar found by the Hills et al. method,” store the exact method, objective, task formulation, numerical settings and source. This is a provenance label, **not** a proof that no four-bar satisfies the benchmark's Fréchet metric.

Unplanted or hand-drawn targets may be included as a separate exploratory set, but they must not be mixed into the primary paired pass/fail analysis unless their ground-truth status and scoring treatment are preregistered.

### 2.6 T1 and T2 development tasks

Use separate, public development tasks for tuning P0 and validating checkers.

- T1: target paths generated from known four-bar mechanisms plus selected textbook examples.
- T2: known input/output angle relationships with declared input ranges and branch constraints.
- Keep all development tasks, target curves and published examples out of the sealed 200-task T3 test set.

### 2.7 T3 cost ordering

For accepted candidates use a preregistered lexicographic cost:

1. Number of links (including ground).
2. Number of joints.
3. Ordered path/function error.
4. Required geometric envelope or footprint, if defined for that task.
5. Estimated slicer time and filament, only for T4.

The primary binary outcome is whether a task receives at least one independently verified feasible candidate within budget. The best feasible cost is a secondary outcome. Do not combine these dimensions into an opaque weighted score after seeing the results.

---

## 3. Process-plan representation (the Matter-Compiler-inspired transferable layer)

Build a process-plan format only for the selected mechanism/FDM workflow. Do not design a universal manufacturing IR.

### 3.1 Required objects

Implement typed records for:

- `TaskSpec`: target path/function, units, tolerances, allowed envelope and task constraints.
- `TopologySpec`: canonical graph, link count, joints, ground/input/output choices and graph hash.
- `MechanismCandidate`: dimensions, pivot coordinates, coupler-point location, assembly branch and provenance.
- `ProcessPlan`: ordered operation DAG, material, machine profile, print settings, assembly and inspection steps.
- `Operation`: operation ID, operation type, inputs, outputs, preconditions, postconditions, parameters and source records.
- `EvidenceClaim`: claim text, evidence level, checker identity/version/hash, inputs/outputs, assumptions and result.
- `BudgetRecord`: wall time, CPU time, memory, topology attempts, objective evaluations, solver calls and LLM usage.

### 3.2 Operation fields

Every operation must have:

- Stable operation ID.
- Declared operation type from a closed initial vocabulary.
- Input artifact IDs and expected types.
- Output artifact IDs and types.
- Preconditions.
- Postconditions.
- Parameters with units.
- Numeric-source records for every engineering number.
- Dependency IDs.
- Checker IDs required before the operation can be accepted.
- Failure modes and recovery action (or explicit `no_recovery`).

### 3.3 Initial operation vocabulary

Keep the vocabulary small and process-specific:

- `define_material`
- `define_machine_profile`
- `create_link_geometry`
- `create_joint_geometry`
- `assemble_mechanism`
- `check_kinematics`
- `check_collision`
- `orient_parts`
- `slice_model`
- `inspect_gcode`
- `assemble_printed_parts` (planned operation only; no physical execution in core scope)
- `inspect_dimensions` (planned operation only; requires L5 physical data)

### 3.4 Plan validation semantics

- Operations form a directed acyclic graph.
- Every dependency references an existing operation.
- Every precondition is established by an earlier dependency or by the task input.
- Every postcondition has a checker.
- Units are explicit and checked (for example with Pint).
- No untyped numeric literal may control a manufacturing rule.
- Every numeric parameter has a source or an explicit label such as `assumption`, `design_guide_default`, `solver_calibrated`, or `measured`.
- A plan cannot claim an operation was physically executed if it was only simulated or compiled.

### 3.5 Example plan fragment

```json
{
  "plan_id": "matter001-task-0007-plan-001",
  "process": "fdm_pla",
  "material": {"name": "PLA", "source_id": "material-source-001"},
  "machine_profile": "profile-pla-fdm-default-v1",
  "operations": [
    {
      "id": "op-create-links",
      "type": "create_link_geometry",
      "inputs": ["mechanism-candidate-001"],
      "outputs": ["link-solids-001"],
      "depends_on": [],
      "parameters": {"units": "mm"},
      "preconditions": ["candidate_passes_v2"],
      "postconditions": ["link_solids_valid"],
      "checker_ids": ["V3-geometry"]
    },
    {
      "id": "op-slice",
      "type": "slice_model",
      "inputs": ["link-solids-001"],
      "outputs": ["gcode-001", "slice-report-001"],
      "depends_on": ["op-create-links"],
      "parameters": {"profile": "profile-pla-fdm-default-v1"},
      "preconditions": ["geometry_passes_v3", "rules_pass_v4"],
      "postconditions": ["slicer_completed_without_warnings"],
      "checker_ids": ["V4-slicer"]
    }
  ]
}
```

The example is a schema illustration, not an approved machine profile or a physically executed manufacturing plan.

---

## 4. Checker V1 — static process-plan validator

### 4.1 Implementation

Use Pydantic models (or an equivalently strict typed schema) for input validation and Pint or an equivalent units library for dimensional checks.

Implement:

1. Schema validation.
2. Closed-vocabulary operation types.
3. Unique operation/artifact IDs.
4. Dependency existence.
5. DAG cycle detection and topological ordering.
6. Preconditions satisfied by declared prior outputs or task inputs.
7. Postconditions linked to checker IDs.
8. Unit compatibility.
9. Numeric source/provenance validation.
10. Prohibition on physical-execution claims without L5 evidence.

### 4.2 Required negative test battery

Create at least 200 deliberately broken plans, including:

- Dependency cycle.
- Missing dependency.
- Duplicate operation ID.
- Unknown operation type.
- Missing input artifact.
- Missing precondition.
- Precondition produced only by a later operation.
- Dimensionally inconsistent units.
- Unitless length or angle.
- Invalid negative dimension.
- Missing source for a numeric rule.
- Unsupported machine profile.
- Postcondition without a checker.
- Attempt to mark a simulated operation as physically executed.

The validator must reject every seeded invalid plan. Also include valid plans to ensure the checker is not simply rejecting everything.

### 4.3 Acceptance

V1 passes only if all valid fixtures pass, all 200 seeded invalid fixtures fail for the expected reason, and property-based tests find no unhandled invalid structure. Record the seed and exact invalidation reason for every case.

---

## 5. Checker V2 — mechanism kinematics

V2 is the critical Phase 0 checker. Implement four-bars first; only then add 6-, 8- and 10-link loop solvers.

### 5.1 Four-bar representation

Use four link lengths and a declared inversion:

- Ground link.
- Input crank/link.
- Coupler.
- Output rocker/link.
- Fixed pivot coordinates.
- Coupler-point coordinates in the coupler frame.
- Assembly branch identifier.

Use one canonical coordinate convention and document angle direction, handedness and units.

### 5.2 Grashof and inversion checks

Let `s` be the shortest link, `l` the longest, and `p`, `q` the remaining links.

For a full-rotation input design require a **strict** Grashof inequality with a declared margin:

` s + l < p + q - delta_g `

`delta_g` must be defined from an absolute and/or relative tolerance and recorded in the task/configuration. Do not accept equality as a robust full-rotation design; equality is a change-point case and is rejected from the initial full-crank benchmark.

Handle valid inversions correctly:

- Shortest link grounded: double-crank/drag-link inversion; the adjacent links can rotate fully.
- Shortest link adjacent to ground: crank-rocker inversion; the shortest moving link can be the full-rotation input crank.
- Longest link grounded: typically double-rocker for the Grashof case; it does not satisfy a full-rotation input-crank requirement and should be excluded from tasks that explicitly require a full crank, not from all possible finite-range tasks.

Do not hard-code a single “shortest link is the input” rule that rejects double-cranks.

### 5.3 Closed-form position analysis

For four-bars, use circle-intersection geometry (or an equivalent standard closed-form formulation) to solve the moving pivots at each input angle. Retain both assembly branches and the signed square-root term that distinguishes them.

Required checks:

- Reachability of the circle intersection.
- Continuity of the selected assembly circuit.
- Correct branch identity across the input trajectory.
- Input-drive dead-centre/toggle conditions.
- Output dead-centre positions classified separately; normal output limit positions of a crank-rocker are not automatically invalid.
- Correct input/output direction and target ordering.

For four-bars, use the algebraic sign/discriminant analysis and closed-form toggle angles. Do not impose general interval arithmetic merely to re-check conditions that have a direct closed-form solution.

A branch/circuit test must not confuse a normal output dead-centre with a forbidden input-drive lock or a change of assembly circuit.

### 5.4 Same-circuit condition

Use the sign of the appropriate square-root/cross-product term under the declared coordinate convention to track the assembly branch. Check that the target positions belong to the same continuous assembly circuit and that the mechanism can traverse the required sequence without disassembly.

The sign convention must be covered by mirrored and relabelled golden fixtures. A sign test without coordinate-convention tests is not sufficient.

### 5.5 Higher-link kinematics

For 6-, 8- and 10-link mechanisms:

- Form loop-closure equations from the canonical graph.
- Enumerate or track feasible assembly modes.
- Use continuation and numerical root solving for candidate generation.
- Detect singularities, branch changes and failed continuation.
- Use root isolation or certified methods only where the specific claim requires it and the method is available.
- If a higher-link checker is numerical rather than certified, label the result accordingly; do not claim a formal proof of continuous feasibility.

Use independent implementations where practical: one solver for candidate generation and a separate loop-closure checker for adjudication.

### 5.6 T1 path metric

Replace point-to-curve-only error with **discrete Fréchet distance** between:

1. The ordered target path.
2. The ordered portion of the coupler curve actually traversed from the first target event to the last target event.

The return stroke after the final target event is unconstrained, as specified by the task.

The metric must preserve order. The solver may not pass near the right points in the wrong order and still pass.

Pre-register:

- Target-point ordering.
- Sampling density and adaptive refinement rule.
- Coordinate units.
- Acceptance tolerance.
- Treatment of repeated/near-identical target points.
- Whether curve orientation may be reversed (default: no, unless the task explicitly allows it).

For final reported error, retain the target and generated samples, the discrete Fréchet result and the sampling-resolution metadata. Do not call the discrete metric a continuous certificate.

### 5.7 T2 function metric

For function-generation tasks:

- Evaluate the declared Freudenstein relationship over the complete specified input range.
- Track the selected assembly branch.
- Record maximum and RMS angular error, with angles wrapped consistently.
- Reject unreachable input regions and branch changes.
- Use the task's preregistered tolerance; do not tune it after seeing proposer results.

### 5.8 V2 self-tests

- Textbook worked examples with published dimensions/results.
- Roberts–Chebyshev cognate test: the three cognate four-bars must reproduce the same coupler curve within the declared numerical/sampling tolerance.
- Independent numerical loop-closure solver versus closed form across random feasible four-bars.
- Mirrored and relabelled mechanism tests.
- Strict Grashof pass/fail boundary tests.
- Equality/change-point rejection tests.
- Double-crank acceptance tests.
- Crank-rocker acceptance tests.
- Normal output dead-centre acceptance tests.
- Input lock/toggle rejection tests.
- Same-circuit and wrong-branch rejection tests.
- Target-order violation tests.
- Unreachable circle-intersection tests.
- Degenerate link/pivot tests.

Phase 0 cannot pass until these tests have documented expected outcomes and independent comparison results.

---

## 6. Checker V3 — CAD geometry and motion clearance

### 6.1 Geometry generation

Generate parametric link and joint solids with CadQuery/OpenCASCADE or build123d backed by an agreed geometry kernel. Save both editable source parameters and neutral output formats (STEP plus STL/3MF as appropriate). Hash every generated artifact.

### 6.2 Static geometry checks

- Every solid is valid in the geometry kernel.
- Meshes are watertight and manifold where a closed solid is expected.
- No self-intersecting or zero-volume solids.
- Minimum wall thickness is above the declared process rule.
- Hole and pin dimensions match the design specification.
- Shape comparison uses **symmetric Hausdorff distance**: maximum of the two directed distances, not only one direction.
- If tessellated meshes are used, record mesh resolution/deflection and do not imply the mesh metric is exact CAD geometry.

### 6.3 Motion and collision checks

Do not rely on a fixed 360-position collision sample alone.

Use the conservative displacement-bound method:

1. For every interval between adjacent mechanism samples, bound the maximum displacement `d` of any point on each moving part from its sampled pose over the interval.
2. Derive the bound from link geometry, translational motion and angular displacement. For a point at radius `r` from a rotation centre, the rotational displacement bound can use `2*r*sin(|delta_theta|/2)`; include any translation of the rotation centre.
3. Let the required clearance be `c_min`.
4. Require sampled clearance at the reference pose to be at least `c_min + 2*d` when each of the two potentially approaching bodies has displacement bounded by `d` over that interval. If the two bodies have different bounds, use `c_min + d_A + d_B`.
5. Refine the sample interval where angular velocity ratios or displacement bounds grow.
6. If the bound cannot be established, mark the interval unresolved; do not silently accept it.

This provides a conservative sampled bound under the stated kinematic displacement assumptions. It is not a generic swept-volume proof.

### 6.4 Print-in-place joints and pin capture

Add explicit tests for:

- Minimum moving clearance throughout the motion cycle.
- Pin/head or retaining-feature dimensions larger than the escape aperture by the declared capture margin.
- No rigid-body escape path through the joint opening under the declared assembly model.
- No collision between captured features over the motion cycle.
- Print orientation does not remove the retaining feature or create an unsupported geometry that violates V4.

A geometric pin-capture check establishes geometric retention only. It does not establish strength, fatigue life or resistance to deformation; those require physics/physical evidence.

### 6.5 V3 tests

- Known watertight and intentionally invalid solids.
- One-sided versus symmetric Hausdorff regression case.
- Known-clearance and known-collision mechanisms.
- Collision between samples that the conservative bound must detect or mark unresolved.
- Pin that is visibly captured versus pin that can escape.
- Mesh-refinement convergence for shape and clearance metrics.
- Geometry round-trip tests from parameter model to STEP/mesh.

---

## 7. Checker V4 — FDM/PLA manufacturing rules and slicer

### 7.1 Rule file

Create one versioned `pla_fdm_rules.yaml` file. Every rule includes:

- Rule ID.
- Parameter and unit.
- Allowed range/condition.
- Source type: manufacturer specification, published design guide, measured calibration, or project assumption.
- Source identifier and version/date.
- Evidence level.
- Whether the rule is hard or advisory.

Initial rules include:

- Build-volume bounds.
- Minimum feature size.
- Minimum hole diameter.
- Minimum wall thickness.
- Overhang threshold (the proposed 45-degree default is a design-guide assumption, not a universal physical law).
- Maximum bridge length.
- Joint clearance range.
- Layer height, nozzle diameter and extrusion-width compatibility.
- Part orientation constraints.

Do not mix a default design-guide value with a measured machine-specific calibration.

### 7.2 Slicer integration

Use an existing slicer CLI (for example, PrusaSlicer) rather than writing a toolpath generator.

Pipeline:

1. Validate geometry.
2. Select a pinned machine/material profile.
3. Generate 3MF/STL input.
4. Invoke slicer with pinned settings.
5. Capture G-code, slicer report, warnings and estimated time/filament.
6. Parse warnings and errors.
7. Reject any output with a disallowed warning or unresolved profile mismatch.
8. Hash the input, profile, slicer version, output G-code and report.

The slicer estimate is not measured print time or measured material usage.

### 7.3 T4 acceptance

A T4 candidate passes only if:

- V2 motion passes.
- V3 geometry and full-cycle clearance pass.
- Pin-capture checks pass where applicable.
- V4 rules pass.
- Slicer completes under the pinned profile without disallowed warnings.

---

## 8. Checker V5 — physics simulation

V5 is optional unless a task has a declared load or flexible-joint requirement.

### 8.1 Solver

Use FEniCSx or CalculiX, selecting one as the initial supported solver. Pin version, solver settings, element types, material model and boundary-condition conventions.

### 8.2 Validation gate

Before any V5 output counts as L3:

1. Reproduce Euler–Bernoulli cantilever-beam deflection within 1% under the stated assumptions.
2. Reproduce one additional standard benchmark appropriate to the solver formulation (for example, a published Cook's membrane reference or another documented benchmark with reference values).
3. Demonstrate mesh convergence for the quantities used in the claim.
4. Check boundary conditions and load application independently.
5. Record solver residuals and numerical tolerances.

A benchmark pass validates the solver setup for the tested class; it does not validate every material model or every printed linkage.

### 8.3 Material assumptions

PLA datasheet values are assumptions unless measured data directly supports the chosen material, print process, orientation and conditions. Account for anisotropy when relevant. Do not promote generic bulk-PLA properties to validated printed-part properties.

---

## 9. Checker V6 — model-conditional robustness

### 9.1 Scope

V6 estimates robustness under an explicitly declared perturbation model. Its output must be called **model-conditional robustness**, not actual manufacturing reliability.

### 9.2 Perturbations

Potential inputs include:

- Link-length dimensional bias.
- Pivot-location errors.
- Hole and pin dimension errors.
- Joint play/clearance.
- Print-orientation-dependent dimensional shifts, if supported by data.

For each perturbation parameter record its distribution, source, evidence level, correlation assumptions and units.

### 9.3 Outputs

- Estimated probability that the design remains within path tolerance under the model.
- Confidence interval from Monte Carlo sampling.
- Worst-case local sensitivity of path error to each design parameter.
- Failure-mode breakdown (collision, closure failure, path error, pin escape, etc.).
- Random seed and sample count.

### 9.4 Data audit

Do not use NIST AMB2022-06 as a published PLA variation dataset if its measurement results are not publicly available/reliable for this purpose. Do not infer PLA joint-clearance distributions from unrelated polycarbonate measurements.

The Mendeley PLA dataset metadata describes 16 specimens in a Taguchi L16 design across print settings, with dimensional deviation, roughness and hardness measurements. Before using it for a stochastic noise model, download and inspect the raw spreadsheets:

- Determine whether there are repeated specimens under identical settings.
- Separate repeated instrument readings from independent printed specimens.
- Check specimen IDs and treatment combinations.
- Identify measurement resolution and instrument uncertainty.
- Determine whether the dimensional measurement locations match linkage features (holes, pins, shafts, wall thicknesses).
- Check whether the printer/material/settings resemble the intended process profile.

If there is one specimen per setting, the dataset may support exploratory factor-response or bias analysis, but it does **not** estimate within-setting print-to-print variability. Six photographs of one specimen are not six independent specimens. In that case, V6 must use labelled design-guide assumptions and remain model-conditional; no L4 manufacturing-variation claim is allowed.

L4 is claim-specific and requires independent published measured data that match the physical property being claimed. If no suitable dataset exists, report “L4 evidence unavailable.” Do not manufacture an L4 label merely to complete the architecture.

---

## 10. Proposers and fair comparison

### 10.1 P0 — classical topology selection

P0 is the strongest frozen conventional baseline, not a deliberately weak first-fit search.

Candidate selection policies to test on the development set:

- Increasing link count.
- Fixed deterministic ordering within each link count.
- Structural-heuristic ranking.
- Budget-aware ranking based on measured optimisation cost from Phase 0.
- Seeded random ordering as a simple baseline.

P0 also uses the common geometry/parameter optimiser and checker stack. For four-bar subproblems, include a specialised polynomial/homotopy method where the task formulation matches its objective. Hills et al. (2024) compute a near-complete numerical set of critical points for a particular polynomial approximate-synthesis objective and report estimates/trace-test results for their formulations. Do not interpret that as a certified lower bound for the benchmark's discrete Fréchet metric, and do not claim it proves that no four-bar exists for a task.

Tune and select the strongest P0 policy on the development set only. Freeze its policy, code hash, parameters and random seeds before hidden testing.

### 10.2 P1 — LLM-guided topology selection

P1 may:

- Inspect the task specification.
- Inspect the same public topology catalogue and topology metadata available to P0.
- Rank or select the next topology/inversion to try.
- Explain a selection in a structured rationale field.
- Use prior *publicly available run feedback* from its own evaluated candidates.

P1 may not:

- See hidden ground-truth topology/dimensions.
- Change the acceptance rules or task tolerance.
- Use a different optimiser or extra objective evaluations.
- Receive more solver calls than P0.
- Submit a candidate directly to the referee without going through the common optimiser/validation pipeline.

Require a strict JSON output schema containing topology IDs from the supplied catalogue. Reject unknown IDs, malformed output and duplicate selections.

### 10.3 P2 — LLM alone

P2 is an ablation. It proposes topology and dimensions without invoking the conventional numerical optimiser. It uses the same checkers and the same acceptance threshold. Report it separately; do not use it as the primary comparison if the primary research question is whether LLM-guided *selection* improves classical search.

### 10.4 Shared downstream optimisation

For every topology selected by P0 or P1:

- Use the same optimiser implementation/version.
- Use the same objective and constraints.
- Use the same initialisation policy and restart count.
- Use the same per-topology candidate-evaluation cap.
- Use the same V2/V3/V4 checkers.
- Record all calls and failures.

This isolates the contribution of topology ordering. If P1 is later allowed to propose continuous starting points or structural edits, that must be a separately labelled experiment.

---

## 11. Budget accounting and multi-budget evaluation

### 11.1 Budget vector

Do not compress all costs into one hidden scalar. Record:

- Topology candidates selected.
- Topology/inversion attempts.
- Objective/constraint evaluations.
- Optimiser runs and restarts.
- Kinematics checker calls.
- Geometry checker calls.
- CPU seconds.
- Wall-clock seconds.
- Peak memory.
- LLM input/output/billed tokens where exposed.
- LLM inference time and monetary cost.
- External model/version and prompt hash.

Internal reasoning tokens that a provider does not expose must not be invented. Report observable token counts and cost.

### 11.2 Equal-budget definition

The principal matched budget is the **downstream search budget**:

- P0 and P1 receive identical candidate-evaluation credits.
- They use the same per-topology optimiser cap.
- They receive identical total solver/checker-call ceilings.
- They receive identical CPU/wall-clock ceilings for the shared optimiser stage.

P1's LLM inference overhead is measured and reported separately, not treated as free. Include an all-in cost comparison (optimiser cost + LLM cost/time) so the paper cannot claim practical efficiency while hiding the additional reasoning expense.

### 11.3 Multiple budget points

After Phase 0 profiles the actual cost of topology evaluations, freeze four or five budget points before hidden testing. Express them in common objective-evaluation credits and publish the corresponding wall-clock/CPU limits. A suitable design is a progression from small to large budgets, ending with the budget required to evaluate every catalogue entry under the common per-topology optimiser cap.

Do not choose budget points after seeing P1's results.

At the exhaustive endpoint, if both arms have evaluated every expanded topology/inversion with the same deterministic optimiser schedule, their best-known result should converge to the same candidate set. If not, investigate implementation or accounting differences. Exhaustion of the topology list does not prove global optimality of the continuous dimensions unless the optimiser itself has a completeness certificate.

### 11.4 Primary and secondary outcomes

- Primary: paired task-level success at one preselected realistic budget `B_primary`.
- Secondary: success-vs-budget curves at all frozen budgets.
- Secondary: best feasible lexicographic cost at each budget.
- Secondary: all-in compute/time/cost.
- Secondary: topology diversity and the number of unique topology families explored.

Choose `B_primary` after Phase 0 profiling but before the hidden test is opened.

---

## 12. Statistical protocol and preregistration

### 12.1 Sample size

Use 200 hidden T3 tasks rather than 50.

The supplied power assumptions are:

- 25% of task pairs are discordant.
- Among discordant pairs, P1-favouring outcomes occur 3:1.
- Two-sided alpha = 0.05.

Under these assumptions, an exact paired-binomial/McNemar calculation gives approximately:

| Tasks | Approximate power |
|---:|---:|
| 50 | 32% |
| 100 | 65% |
| 150 | 85% |
| 200 | 94% |

These values were independently reproduced for the stated assumptions. They are not guaranteed power: the realised discordance rate and split may differ. Publish a sensitivity table for plausible alternatives before the test. Power for paired binary outcomes depends on the discordant-pair rate, not merely the total task count.

### 12.2 Paired outcome definition

For each hidden task, P0 and P1 receive the same task and budget. Each arm's outcome is:

`success = at least one independently verified feasible design returned within the task's fixed total budget`.

If multiple restarts are used, give both arms the same number of restarts and the same total downstream evaluation budget; split that budget according to a preregistered schedule. Do not count repeated runs as independent tasks.

### 12.3 Primary statistical test

Use two-sided exact McNemar's test at alpha 0.05 on the 200 paired task outcomes at `B_primary`.

Report:

- P0-only successes.
- P1-only successes.
- Both-success tasks.
- Neither-success tasks.
- Paired success-rate difference.
- Confidence interval for the paired difference.
- Exact McNemar p-value.
- All task-level outcomes.

The primary success claim requires both a positive paired effect and the preregistered statistical criterion. A p-value alone is not a practical-effect claim.

### 12.4 Multiple budgets and secondary analysis

Show all frozen success-vs-budget curves. Do not run a separate uncorrected significance test at every budget and then select the best-looking one. Treat non-primary budget points as descriptive or apply a preregistered multiplicity method (for example, Holm correction) if formal secondary hypothesis tests are required.

For the best-cost metric, report paired task-level differences and confidence intervals. Use a paired bootstrap over tasks or another preregistered method; preserve task pairing.

### 12.5 Replication

After the primary hidden test, repeat on a second independently generated hidden task set or a separately preregistered model/seed block. Keep the primary result unchanged. If using repeated stochastic runs on the same tasks, account for task clustering; do not pretend those runs are independent new tasks.

### 12.6 Preregistration freeze

Before opening hidden tasks, freeze and hash:

- Task generator and hidden seeds (store seeds sealed).
- Topology catalogue and canonicalisation code.
- P0 policy and tuning results.
- P1 model/version, prompt template and output schema.
- P2 configuration.
- Optimiser and checker versions.
- All thresholds and cost ordering.
- Budget vector and budget points.
- Primary budget.
- Primary outcome and statistical test.
- Power assumptions and sensitivity analysis.
- Exclusion rules (preferably none after unblinding; all exclusions must be pre-specified).

---

## 13. Evidence levels and APEDP-X adjudication

Evidence is attached to **individual claims**, not granted wholesale to a design or project.

| Level | Evidence | Permitted wording |
|---|---|---|
| L1 | Typed process plan and encoded-rule checks pass | “The plan passes the declared static rules.” |
| L2 | Geometry-kernel checks and the declared kinematic/shape checks pass | “The CAD geometry/motion model passes the specified software checks within stated tolerances.” |
| L3 | Physics solver passes its validation benchmarks and the specific simulation passes | “The benchmark-validated simulation predicts … under the stated model.” |
| L4 | Independent published measured data directly relevant to the claim, with held-out comparison | “The prediction agrees with the specified published measurements within the reported uncertainty.” |
| L5 | The physical device is built and measured | “The built specimen was measured/tested under the recorded conditions.” |

Rules:

1. Never describe L1-L4 as proof that a physical item was built.
2. Never promote a claim beyond the strongest checker/data source that actually supports that claim.
3. A geometry claim may be L2 while a strength claim for the same candidate is only L1/unvalidated.
4. L4 requires relevance of material, process, geometry/property and measurement method; a vaguely related dataset is not enough.
5. L5 is out of scope unless a later physical phase is explicitly authorised and actually performed.
6. Any geometry change invalidates downstream geometry, slicer, physics and robustness claims; rerun affected checkers.
7. Any rule, solver, model or dataset change creates a new evidence record and invalidates dependent claims.

### 13.1 Evidence record fields

Every claim stores:

- Claim ID and exact claim text.
- Evidence level.
- Task and candidate IDs.
- Checker name/version/source hash.
- Input artifact hashes.
- Output artifact hashes.
- Configuration and environment.
- Random seeds.
- Assumptions and their sources.
- Numeric tolerance and units.
- Result/status and failure diagnostics.
- Parent claims on which the claim depends.
- Timestamp and immutable ledger sequence.

APEDP-X decides VALID/INVALID/UNRESOLVED for the declared claim and its evidence. The proposer cannot promote its own output.

---

## 14. Repository structure and module boundaries

Suggested initial repository layout:

```text
mmlm_matter_001/
  README.md
  SPEC_MMLM_MATTER_001.md
  preregistration/
    protocol.yaml
    protocol.sha256
    hidden_task_manifest.sealed
  configs/
    environment.lock
    fdm_pla_rules.yaml
    machine_profiles/
  schemas/
    task_spec.py
    topology_spec.py
    mechanism_candidate.py
    process_plan.py
    evidence_claim.py
    budget_record.py
  topology/
    enumerate_chains.py
    expand_inversions.py
    canonicalize.py
    mobility.py
    catalogue.py
  tasks/
    generate_planted.py
    generate_development.py
    seal_hidden.py
    validate_task.py
  kinematics/
    fourbar_closed_form.py
    fourbar_branch.py
    multiloop_solver.py
    loop_closure.py
    path_metrics.py
  geometry/
    build_links.py
    build_joints.py
    shape_metrics.py
    collision_bound.py
    pin_capture.py
  manufacturing/
    rules.py
    orientation.py
    slicer_cli.py
    gcode_report.py
  physics/
    fea_adapter.py
    benchmark_cantilever.py
    benchmark_second.py
    mesh_convergence.py
  robustness/
    perturbations.py
    monte_carlo.py
    sensitivity.py
    data_audit.py
  proposers/
    p0_classical.py
    p1_llm_topology.py
    p2_llm_direct.py
  evaluation/
    optimiser.py
    budget_accountant.py
    run_benchmark.py
    budget_curves.py
    paired_stats.py
  evidence/
    claim_ledger.py
    hashes.py
    apedpx_adapter.py
  tests/
    test_schema.py
    test_topology_catalogue.py
    test_fourbar.py
    test_cognates.py
    test_branch_and_toggle.py
    test_path_metrics.py
    test_geometry.py
    test_collision_bound.py
    test_pin_capture.py
    test_slicer.py
    test_physics_benchmarks.py
    test_budget_accounting.py
    test_preregistration_integrity.py
```

Keep module boundaries strict:

- Proposers produce candidates or topology rankings only.
- Optimisers produce candidate parameter sets.
- Checkers produce structured evidence and diagnostics.
- APEDP-X adjudicates claims; it does not search for or repair candidates.
- The ledger records provenance; it is not itself a correctness authority.

Use Python first. Add Rust only if Phase 0 profiling demonstrates a specific hot path that materially limits the benchmark. Do not introduce native complexity pre-emptively.

---

## 15. Phase-by-phase implementation schedule

The planned start is after 20 January 2027. The original estimate is 7–9 weeks, conditional on gates passing. Phase durations are planning estimates, not guarantees.

### Phase 0 — mathematical foundation and cost profiling (1 week)

**Purpose:** establish a trustworthy four-bar checker, topology catalogue, task generator and strong P0 baseline; measure whether exhaustive topology evaluation is actually expensive.

Tasks:

1. Create the repository skeleton and locked environment.
2. Implement canonical link/joint graph types.
3. Implement base-chain catalogue ingestion/enumeration for 4/6/8/10 links.
4. Expand ground/input/output inversions and canonicalise duplicates.
5. Validate catalogue counts against the declared source conventions.
6. Implement four-bar closed-form kinematics.
7. Implement strict Grashof/inversion logic, branch/circuit checks and toggle classification.
8. Implement discrete Fréchet path metric over the traversed path segment.
9. Implement the independent numerical loop-closure reference solver.
10. Implement textbook and Roberts–Chebyshev cognate fixtures.
11. Implement a planted-task generator for development tasks.
12. Implement P0 policies and tune them on development data only.
13. Profile optimisation/evaluation cost for each topology class and inversion.
14. Determine whether the complete expanded catalogue can be searched under practical budgets.
15. Draft budget levels and freeze them before hidden testing.

**Go/no-go gate:**

- All V2 self-tests pass.
- The independent kinematics implementations agree within the frozen tolerance on feasible fixtures.
- P0 finds the preregistered proportion of planted development solutions.
- Topology counts and inversion expansion are reproducible.
- The Phase 0 report includes actual per-topology evaluation cost and whether exhaustive search is feasible.

If any condition fails, repair the checker/catalogue/baseline. Do not start Phase 1 with an untrusted verifier.

### Phase 1 — P0/P1/P2 matched benchmark (2 weeks)

**Purpose:** measure the LLM's contribution to topology selection.

Tasks:

1. Tune P0 policy using development tasks only; select and freeze the strongest policy.
2. Freeze P1 model, prompt, topology-ranking output schema and token/time limits.
3. Freeze P2 direct-design prompt and output schema.
4. Generate and seal 200 hidden T3 tasks, stratified across planted 6/8/10-link mechanisms.
5. Hash all protocol files and hidden task manifest.
6. Run P0, P1 and P2 on identical tasks at all frozen budget points.
7. Use the same optimiser, per-topology restart schedule and checkers for P0/P1.
8. Store every attempted topology, candidate, failure reason and budget consumption.
9. Compute primary exact McNemar test at `B_primary`.
10. Report all success-vs-budget curves and the exhaustive endpoint.
11. Report LLM overhead and all-in cost separately.

**Go/no-go gate:**

- P1 has a positive paired effect over P0 at the preregistered primary budget.
- The primary statistical criterion is met.
- The result is not explained by extra optimiser calls or different checker access.
- The advantage is visible in verified outputs, not only LLM explanations.

If P1 does not beat P0, record the negative result and stop the LLM-specific MMLM-MATTER work. Do not create more architecture to explain away a null result.

### Phase 2 — process plan, geometry and slicer (2 weeks)

**Purpose:** turn accepted mechanisms into valid printable design/process artifacts.

Tasks:

1. Implement V1 process-plan schema and 200-invalid-plan rejection battery.
2. Implement parametric link, joint and captured-pin geometry.
3. Implement mesh validity/manifold/watertight checks.
4. Implement symmetric Hausdorff shape comparison with tessellation metadata.
5. Implement clearance sampling plus conservative per-interval displacement bound.
6. Implement pin-capture and retention geometry checks.
7. Implement FDM/PLA design-rule file with source/evidence tags.
8. Integrate pinned slicer CLI and machine profile.
9. Parse G-code/report/warnings and create a hash-addressed build artifact.
10. Run T4 print-in-place tests on accepted T1/T3 candidates.

**Go/no-go gate:**

- All valid process-plan fixtures pass; all 200 broken fixtures fail for expected reasons.
- Accepted geometry is valid and watertight where expected.
- Clearance checks either pass with a conservative bound or are marked unresolved.
- Captured pins pass the geometric retention rule.
- Slicer completes without disallowed warnings under the pinned profile.

### Phase 3 — physics and robustness (1–2 weeks)

**Purpose:** add carefully bounded L3 simulation and model-conditional robustness.

Tasks:

1. Implement a cantilever benchmark and achieve the 1% reference criterion.
2. Implement a second published solver benchmark.
3. Run mesh-convergence studies.
4. Implement structural/deflection simulation only for tasks with loads.
5. Audit the Mendeley PLA dataset raw files for repeats and relevance.
6. Exclude unsupported NIST AMB2022-06 claims from the PLA noise model.
7. Implement a source-tagged perturbation model.
8. Run Monte Carlo robustness and parameter sensitivity.
9. Keep design-guide-only distributions labelled as assumptions.

**Go/no-go gate:**

- V5 reproduces both benchmarks and has a documented convergence result.
- No material parameter is described as calibrated unless supported by relevant measured data.
- V6 reports model-conditional robustness and uncertainty.
- No L4 status is assigned without a relevant independent dataset and held-out comparison.

### Phase 4 — closed-loop correction in simulation (1 week)

**Purpose:** test whether correction improves performance under a declared perturbation model.

Tasks:

1. Freeze the perturbation/noise model and separate development from held-out random seeds.
2. Inject systematic dimensional bias and random deviations.
3. Simulate measurement outputs under the declared model.
4. Run a correction algorithm that adjusts clearances/dimensions.
5. Compare against a no-correction baseline and a simple deterministic calibration baseline.
6. Evaluate on held-out perturbation seeds and task instances.
7. Record model exploitation checks and sensitivity analyses.

**Go/no-go gate:**

- Correction beats no correction on preregistered held-out simulations.
- It also beats or is compared fairly with the simple calibration baseline.
- The claim remains L3/model-conditional unless supported by matching published measurements (L4).

### Phase 5 — optional physical validation (1 week or longer)

This phase is not required for the software research result and is not assumed in the core schedule.

If authorised later:

1. Use an available printer and a pinned PLA profile.
2. Print a preregistered sample of accepted designs (10 is an initial pilot, not a universal reliability sample).
3. Measure dimensions with calibrated instruments.
4. Measure coupler-point motion using a documented camera/tracking setup.
5. Compare observed results with predictions and uncertainty intervals.
6. Store raw photographs/videos, measurement calibration and specimen IDs.
7. Promote only the specific physical claims supported by the samples to L5.

---

## 16. Required reports and artifacts

At the end of each phase produce:

1. `PHASE_REPORT.md` — objective, method, results, failures, limitations and gate decision.
2. `environment.lock` — exact dependency versions and runtime information.
3. `protocol.yaml` and SHA-256 hash.
4. `topology_catalogue.jsonl` — canonical base chains and expanded inversions.
5. `task_manifest.jsonl` — public development tasks; hidden manifest stored sealed.
6. `checker_results.jsonl` — every check, including failures and unresolved cases.
7. `budget_ledger.jsonl` — per-arm and per-task resource accounting.
8. `evidence_ledger.jsonl` — claim-level L1–L5 records.
9. `candidate_artifacts/` — CAD, STEP, STL/3MF, G-code and reports as applicable.
10. `statistics_report.md` — paired outcome tables, curves, confidence intervals, tests and power assumptions.
11. `SHA256SUMS` — integrity manifest.

No summary table may omit failed, unresolved or excluded candidates. Exclusion requires a preregistered reason.

---

## 17. Main implementation risks and controls

| Risk | Control |
|---|---|
| Checker bug | Known-good and known-bad fixtures, independent implementations, property-based tests, hashes |
| P0 intentionally weak | Tune candidate policies on development data, select strongest, freeze before hidden test |
| P1 receives more compute | Common optimiser/evaluation budget; separately record LLM tokens, time and cost |
| Hidden-task leakage | Seal task seeds/targets; expose only declared task specification |
| Topology duplicate inflation | Graph canonicalisation and explicit inversion identity |
| False four-bar impossibility claim | Use “not found by method X under objective Y/settings Z”; no infeasibility claim without certificate |
| Optimiser exploits path sampling | Discrete Fréchet, ordered traversal, adaptive sampling and independent checks |
| Collision between samples | Conservative displacement bound; unresolved if bound unavailable |
| Print-in-place pin escapes | Explicit capture geometry test plus full-cycle collision/clearance checks |
| Physics solver misuse | Benchmark gate, mesh convergence and source-tagged material assumptions |
| Simulator exploitation | Held-out perturbation seeds, simple baseline, sensitivity analysis and no L4/L5 overclaim |
| Dataset lacks repeats | Raw data audit; do not infer within-setting variance from one specimen per condition |
| Scope creep | One process, one material, one mechanism family; generalise only after evidence |

---

## 18. Freeze checklist before Phase 1

Phase 1 may not begin until all items are checked:

- [ ] Four-bar closed-form checker passes all golden and adversarial tests.
- [ ] Roberts–Chebyshev cognate test passes within the declared tolerance.
- [ ] Independent numerical loop-closure implementation agrees.
- [ ] Strict Grashof margin and all allowed inversions are correctly handled.
- [ ] Input dead-centre and output dead-centre are distinguished.
- [ ] Target path metric is ordered discrete Fréchet on the traversed segment.
- [ ] Topology catalogue conventions and counts are documented.
- [ ] Ground/input/output inversion expansion is complete for the declared universe.
- [ ] P0 policies have been tuned on development data and frozen.
- [ ] P1 model, prompt and output schema are pinned.
- [ ] P0/P1 share optimiser, checker and downstream search budgets.
- [ ] LLM overhead reporting is implemented.
- [ ] 200 hidden tasks are generated, stratified and sealed.
- [ ] Primary budget and secondary budget points are frozen.
- [ ] McNemar test, alpha, outcome definition and power assumptions are preregistered.
- [ ] Multiple-budget analysis and any multiplicity correction are preregistered.
- [ ] Protocol and code hashes are recorded.

---

## 19. Phase 0's concrete deliverables

Phase 0 is deliberately narrow. It should produce evidence that the experiment is executable, not attempt to build the entire system.

**Deliverable A — topology catalogue**

- Canonical base-chain records for 4/6/8/10 links.
- Expanded ground/input/output inversions.
- Duplicate/isomorphism report.
- Enumeration-convention note.
- Profiling report showing actual catalogue size and per-topology optimisation cost.

**Deliverable B — trusted four-bar checker**

- Closed-form position solver.
- Grashof/inversion classifier.
- Assembly-circuit and branch tracking.
- Input/output toggle classification.
- Discrete Fréchet path metric.
- Golden, cognate and adversarial test results.

**Deliverable C — task generator**

- Reproducible planted four-bar development tasks.
- Planted 6/8/10-link task generation prototype.
- Task validity checker and sealed-seed support.

**Deliverable D — frozen P0 baseline**

- Candidate policies evaluated on development data.
- Best policy selected using a preregistered development metric.
- Source/configuration hashes and resource profile.

**Deliverable E — budget feasibility report**

- Cost per topology and inversion.
- Cost per optimiser evaluation.
- Estimated exhaustive-catalogue cost.
- Proposed 4–5 budget points.
- A documented decision on whether exhaustive search is feasible in the intended environment.

If these five deliverables are complete and the V2 gate passes, Phase 1 can begin. If not, remain in Phase 0.

---

## 20. Source notes

1. **Hills, Baskar, Plecnik and Hauenstein (2024), “Computing complete solution sets for approximate four-bar path synthesis,” Mechanism and Machine Theory 196, 105628.** The paper presents a polynomial objective and random-monodromy/parameter-homotopy approach to finding critical points, with estimated completeness and trace-test results for particular formulations. It is a strong numerical reference, not a certified lower bound for the benchmark's discrete Fréchet metric. DOI: `10.1016/j.mechmachtheory.2024.105628`.
2. **Planar kinematic-chain enumeration reference.** The cited table reports 1, 2, 16 and 230 useful unique one-DOF chains for 4, 6, 8 and 10 links respectively, while noting that unique inversions and certain joint classes are not included. The project must state and preserve the exact catalogue convention.
3. **Mendeley PLA dataset, version 2.** DOI: `10.17632/mf76r4zxg4.2`. Its public description reports 16 PLA specimens under a Taguchi L16 design with dimensional deviation, surface roughness and Shore D hardness measurements. The raw spreadsheet must be audited for independent repeats before estimating within-setting manufacturing variance.
4. **McNemar power.** Power depends on the discordant-pair rate and the split between P0-only and P1-only successes. The 200-task estimate is conditional on the preregistered 25% discordance and 3:1 favourable split; reproduce the calculation in code and provide sensitivity analysis.

Useful links:

- Hills et al. publisher record: https://www.sciencedirect.com/science/article/pii/S0094114X24000557
- Hills et al. preprint: https://www3.nd.edu/~jhauenst/preprints/bhphApproxSynthesis.pdf
- Kinematic-chain enumeration table: https://www.mdpi.com/2411-9660/10/3/46
- Mendeley PLA dataset v2: https://data.mendeley.com/datasets/mf76r4zxg4/2
- McNemar power reference: https://cttir.github.io/tutorials/tutorials/sample-size/power-mcnemar.html

---

## Final implementation principle

Do not build a general Matter Compiler first. Build one trustworthy mechanism-synthesis and process-compilation path, measure whether LLM topology selection adds value over a strong conventional baseline, and promote only claims supported by the corresponding checker and evidence level.

The central experiment is not whether an LLM can describe a mechanism. It is whether, under a frozen and fair protocol, it can repeatedly select better mathematical structures that lead to independently verified, lower-cost engineering designs.
