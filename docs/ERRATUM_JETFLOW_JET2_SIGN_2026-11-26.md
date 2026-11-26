# Erratum: JetFlow's Jet2 step had its curvature reversed from 6 to 26 November 2026

**What was wrong.** `JetFlowStepper.compute_acceleration_torch`
(`packages/jetflow-video/jetflow_video/stepper.py`) passed its two velocity samples to
`_acceleration_between(v_from, v_to, h)` the wrong way round, in both its one-sided and its central form,
so the second-order step z + dt v + dt^2/2 a used a = -dv/dt. On dz/dt = z, one step from t = 1 to 0
landed at -0.5 z, where the exact answer is z / e (0.37 z), the corrected step gives 0.5 z and Euler gives 0:
worse than Euler. The independent review of PREREG-VIDEO-FLOW-2026-11 found it on 26 November.

**When.** The swap came in with commit 99f15f2 (6 November), which moved the quotient into
`_acceleration_between` for the native kernel; before it, the step computed (v_probe - v_current) / h,
which is right. Fixed on the owner's word of 26 November (21:56, "fix the bug in jetflow"), by passing the
samples in the right order; the native kernel itself (`rust/apedp-kernel/src/jetflow.rs`) was right
throughout. `tests/test_jetflow_stepper_curvature.py` pins the fix: one step on dz/dt = z gives 0.5 z in
both forms, dz/dt = t is integrated exactly forward and backward in time, and the error falls by about a
quarter each time the steps double.

**What it touched.** No recorded result. Every recorded JetFlow benchmark that uses the second-order step
predates the swap: `benchmarks/jetflow_vs_diffusion_matched_results.json`, its collision version,
`benchmarks/jetflow_video_v03_confirmation_results.json`, `sharp_wave_upsample_results.json`,
`sharp_wave_pipeline_results.json` and `jet_flow_video_benchmark_results.json` (24 October), and
`benchmarks/jetflow_integrator_sweep.json` (recorded before the swap the same day). The last was checked:
rerun with the fixed step it reproduces its record to every digit; rerun with the reversed step its first
row's Jet2 error would be 0.898 instead of 0.165. So the README's comparison of JetFlow with diffusion (a
tie on the fluid clips, a loss on the collision clips) stands. What the reversed step did touch: the
rehearsals of PREREG-VIDEO-FLOW-2026-11 on 26 November, which are development runs and say so; that
plan's final run uses the fixed step.
