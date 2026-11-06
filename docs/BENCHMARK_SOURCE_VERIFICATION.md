# Do the two experimental benchmark tables match their cited sources?

**Verdict: both are marked `unverified`.** No accessible public source contains
either table. Ten of the eleven copper rows have no traceable record at all, the
one that does sits 7% above the strongest published value at the same grain size,
the NACA 0012 values disagree with the closest measurement by a constant 47%, and
the provenance line on each table names a report family rather than a table.

The measurement is in `benchmarks/external_benchmarks_source_check.json`, produced
by `scripts/verify_external_benchmark_sources.py`, and the two benchmark classes
report it as `provenance()["verified"] = False`, so a caller of
`run_benchmark_audit` can see the status without opening this page. **Nothing was
changed in the tables**: a number that cannot be traced is marked, not silently
adjusted. Inventing a correction would be a larger defect than the one being
reported.

Re-run with `python scripts/verify_external_benchmark_sources.py`. Exit `0` means
every source was reachable and the recorded comparison still holds, `1` means a
measurement moved, `2` means a source could not be fetched — and an unreachable
source is recorded as unverified, never as verified.

## Sources actually fetched

| key | citation | URL | bytes | SHA-256 |
|---|---|---|---|---|
| `harris_tm81927` | Harris, NASA TM-81927, 1981 (NACA 0012, Langley 8-foot TPT) | `ntrs.nasa.gov/api/citations/19810014503/downloads/19810014503.pdf` | 5,054,434 | `9ace5d185c15…` |
| `ladson_cp` | Ladson, Hill & Johnson, NASA TM 100526, 1987 (digitized by NASA TMR) | `tmbwg.github.io/turbmodels/NACA0012_validation/CP_Ladson.dat` | 6,331 | `547e022634a3…` |
| `gregory_cp` | Gregory & O'Reilly, R&M 3726, 1970 (digitized by NASA TMR) | same directory, `CP_Gregory_expdata.dat` | 1,608 | `38d8474bc676…` |
| `meyers_review` | Meyers, Mishra & Benson, Prog. Mater. Sci. 51 (2006) 427–556 | `meyersgroup.ucsd.edu/papers/journals/Meyers 263.pdf` | 5,911,838 | `af1c77b2681d…` |

Full hashes, HTTP status and byte counts are in the record. AGARD-AR-138 itself
was **not** fetched — it is not available at an accessible public URL — so the
AGARD Cp distributions remain unchecked, and the comparison below uses the two
NASA sources that are digitized and machine-readable.

## NACA 0012: Cp at `x/c = 0.25` across Mach

**What the cited report contains.** The 1981 Langley report has one table, and it
is the airfoil's section coordinates (`TABLE I.- SECTION COORDINATES OF NACA 0012
SYMMETRICAL AIRFOIL`). Its chordwise pressure distributions, including the
Mach-number effects the table claims to use, are **figures**. There is no table of
Cp at a fixed chord station against Mach.

**What the accessible measurements give at `x/c = 0.25`.** Interpolated between
the bracketing taps in the digitized data:

| source | condition | Cp(0.25) |
|---|---|---|
| Ladson (TM 100526) | M = 0.30, α = 0.0169°, Re 6e6 | **−0.3551** |
| Ladson | M = 0.30, α = 10.03°, Re 6e6 | −0.9964 |
| Ladson | M = 0.30, α = 15.03°, Re 6e6 | −1.2218 |
| Ladson | M = 0.30, α = 0.0083°, Re 3e6 | −0.3347 |
| Ladson | M = 0.30, α = 0.0210°, Re 9e6 | −0.3762 |
| Gregory (R&M 3726) | α = 0°, Re 2.88e6 | −0.3796 |
| Gregory | α = 10° | −1.1293 |
| Gregory | α = 15° | −1.4335 |

The first thing this shows is that the table **cannot be reproduced without an
angle of attack**: at M = 0.30 the same source gives −0.355 at α ≈ 0 and −0.996
at α = 10°, a factor of 2.8. The benchmark states neither α nor Reynolds number.

**Comparison.** Taking the α ≈ 0 measurement at M = 0.30, recovering the
incompressible value (−0.3387) and scaling it with Prandtl–Glauert, which is the
only way a single Mach tap can be compared with a Mach sweep:

| M | table Cp | PG from measured M=0.30 | table / measurement |
|---|---|---|---|
| 0.50 | −0.577 | −0.391 | **1.475** |
| 0.60 | −0.625 | −0.423 | **1.476** |
| 0.70 | −0.700 | −0.474 | **1.476** |
| 0.75 | −0.756 | −0.512 | **1.476** |
| 0.80 | −0.833 | −0.565 | **1.476** |
| 0.82 | −0.885 | −0.592 | 1.496 |
| 0.85 | −0.980 | −0.643 | 1.524 |
| 0.88 | −1.150 | −0.713 | 1.613 |
| 0.90 | −1.280 | −0.777 | 1.647 |
| 0.92 | −1.340 | −0.864 | 1.551 |
| 0.95 | −1.380 | −1.085 | 1.272 |

Every value from M = 0.50 to M = 0.80 is a **constant 1.475×** the scaled
measurement — to three decimal places, five Mach numbers in a row. Measured
compressibility effects do not have that shape; a series constructed by scaling
one number does. The 47% offset itself would correspond to an angle of attack
around 2–3°, which the table never states, and the ratio then drifts to 1.61–1.65
before falling at 0.95, which no single α would produce.

**Verdict: `unverified`.** Nothing in the accessible literature supports these
eleven numbers as a measured table. What would verify them: pressure-distribution
figures-digitized at a stated α and Re, or the AGARD-AR-138 tables themselves.

## Nanocrystalline copper: yield strength against grain size

**What is actually published.** The accessible review's Table 2 (p. 444–445) gives
compressive yield strengths for inert-gas-condensed Cu, with the hardness-over-three
column that approximates yield when work hardening is small:

| sample | grain size | σ_y | Hv/3 | density |
|---|---|---|---|---|
| Cu1 | 19 nm | **650 MPa** | 770 MPa | 92.5% |
| Cu2 | 20 nm | **850 MPa** | 870 MPa | 98.4% |

The review also states the compiled range: *"They vary from 100 to 900 MPa, for
grain sizes of 100 µm to 26 nm."* It records the sub-25 nm trend as **ambiguous** —
some datasets plateau, others decrease — which is why it lists the inverse
Hall–Petch literature rather than a single turnover point.

**Comparison with the repository table.** The repository's 20 nm row is 910 MPa:

| | value |
|---|---|
| repository, 20 nm | 910 MPa |
| published, 19 nm and 20 nm | 650 / 850 MPa |
| above the strongest published | **+7.1%** |
| above the weakest published | **+40.0%** |

The repository's peak is 980 MPa at 12 nm, which exceeds the review's stated
900 MPa envelope for 100 µm–26 nm. The ten remaining rows (50 µm, 20 µm, 5 µm,
1 µm, 200 nm, 50 nm, 12 nm, 8 nm, 5 nm, 3 nm) have **no traceable record** in the
accessible source. The synthesis route is also different: the benchmark says
electrodeposited micropillars, while the published points are IGC compacts
measured in compression, and the class name says "microcompression".

**Verdict: `unverified`.** One point is comparable and sits 7% outside the
published range at the same grain size; the rest are untraceable. What would
verify them: a per-grain-size table with synthesis route and specimen geometry from
the Greer/Uchic papers themselves, which are not openly accessible.

## What this check does not establish

- It does not show the tables are *wrong* as physics. A 12 nm Cu peak is a
  plausible shape and −0.577 at M = 0.50 is a plausible suction peak at some α.
  It shows the numbers are **not traceable to the stated source**, which is the
  only claim a benchmark table can make on its own.
- Prandtl–Glauert is a model, used here to make one measured Mach point
  comparable with a sweep. At M = 0.80 and above it is outside its own validity
  range, so the ratios there are indicative, not corrections.
- AGARD-AR-138 and the Greer/Uchic papers were not fetched. Both verdicts rest on
  the accessible NASA digitizations and one review; a reader with access to the
  primary papers could overturn the copper verdict in particular.

## Where the status is reported

`apedp.ingestion.external_benchmarks.NACA0012TransonicBenchmark.provenance()` and
`NanocrystallineMicrocompressionBenchmark.provenance()` return `verified: False`
with the citation, the check date, the record path and the reason above.
`tests/test_external_benchmark_source_verification.py` pins the record, the two
dataset hashes, and the flag, so a table cannot be quietly presented as measured.
