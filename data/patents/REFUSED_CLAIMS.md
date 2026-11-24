# Claims refused as operand-only

`is_operand_only()` holds a claim apart when its normalised form names its
operands and one operation and nothing about them. Any expression built the same
way is isomorphic to such a form, so a match is a coincidence of form class rather
than evidence that the patent claims the candidate: the engine returns
`CLAIM_STATES_OPERANDS_ONLY` and never `KNOWN_IN_PATENT_CORPUS`.

7 of 15 normalised claims are refused this way, across 4 of the 29 records; every other normalised claim is decisive and can carry a verdict.

## EP4334731B1

3 of 3 normalised claims refused; 0 decisive.

| claim | normalised form | refused because | what the claim states |
|---|---|---|---|
| 2 | `eta_C*i_ch` | bare product — the claim names two operands and their product and says nothing about either, so any product is isomorphic to it | Claim 2 states 'the said efficiency characteristics are defined as: î ch = η C i ch' (net current injected equals charging efficiency times gross current) and the mirror relation for discharging. Claim 1 renders these as image placeholders, so claim 2 was used and the choice is recorded here. A bare product, so the engine treats it as operand-only. |
| 7 | `C_t_I/C_0_I` | bare quotient — the claim names two operands and their quotient and says nothing about either, so any quotient is isomorphic to it | Claim 7 states 'SOH parameter in time is defined as SOH t = C t I / C 0 I ... where C 0 I corresponds with the first determination of mean battery charge capacity C I ... and C t I is a newly determined value C I'. Normalised as the quotient of charge capacities; operand-only for the same reason as claim 2. |
| 8 | `C_t_E/C_0_E` | bare quotient — the claim names two operands and their quotient and says nothing about either, so any quotient is isomorphic to it | Claim 8 states the same state-of-health definition over 'mean battery energy capacity C E': 'SOH t = C t E / C 0 E'. Normalised as the quotient of energy capacities; distinct from claim 7's charge capacity, so both are recorded. |

## US12021452B2

1 of 1 normalised claims refused; 0 decisive.

| claim | normalised form | refused because | what the claim states |
|---|---|---|---|
| 1 | `V_in/V_out` | bare quotient — the claim names two operands and their quotient and says nothing about either, so any quotient is isomorphic to it | Claim 1 states 'a first switched-capacitor (SC) stage configured for a 2:1 conversion ratio', and further cascaded stages 'each having an arbitrary conversion ratio'. Normalised as the ratio of input to output voltage; the direction (input over output) is my reading of 'conversion ratio', and it is recorded here as such. COARSE: the claim names the two quantities and their ratio and no functional form, so any quotient is isomorphic to it and the engine holds it as operand-only. |

## US20250321289A1

1 of 2 normalised claims refused; 1 decisive.

| claim | normalised form | refused because | what the claim states |
|---|---|---|---|
| 11 | `SoH_pred` | bare symbol (an equality of names) — the claim equates a name with a name and states no operation, so any single symbol matches it | Claim 11 sets 'the battery state of health estimation equal to the battery SoH prediction' when the input data matches no known charging scenario. Normalised as the bare prediction symbol: an equality of names with no operation, so it is operand-only and can never decide a verdict. It is recorded because the claim does state a relation, and the count should show what kind of relation it is. |

## US8965721B2

2 of 2 normalised claims refused; 0 decisive.

| claim | normalised form | refused because | what the claim states |
|---|---|---|---|
| 1 | `Z_meas/Z_ref` | bare quotient — the claim names two operands and their quotient and says nothing about either, so any quotient is isomorphic to it | Step (b): 'determining a ratio of said measured DC impedance to a reference DC impedance ... to establish an impedance degradation factor'. COARSE: the claim names the operands but not their functional form, so this expression matches any quotient; the engine holds it as operand-only and refuses to conclude from a match. |
| 8 | `I_lim*V_min` | bare product — the claim names two operands and their product and says nothing about either, so any product is isomorphic to it | Claim 8 states '(f) calculating a power discharge limit by multiplying said current discharge limit by V min'. Normalised as the product; the claim names the operands and the operation, not a functional form beyond the product, so it is operand-only too. |
