# RPA v0 8-state interface diagnosis (2026-09-15)

Frozen states: `state_000`, `state_002`, `state_003`, `state_020`, `state_023`, `state_028`, `state_001`, `state_004`. Deterministic coverage selection was performed before model calls.

GPU job `pt-ot7ymj3s` wrote all 56 requested calls: B ranked joint (8), C unranked joint (8), and D single-action (40). The job failed only during its original aggregation stage after all cache rows were written. Raw rows remain in the project cache; no model calls were repeated.

| condition | new calls | action coverage | complete outputs | bearing | framing | goal | progress | collision | successor consistency | grounded consistency | ranking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A historical joint | 0 | 32/40 (.800) | 0/8 | 3/40 (.075) | 20/40 (.500) | 31/40 (.775) | 8/40 (.200) | 32/40 (.800) | 48/120 (.400) | 21/120 (.175) | 1/8 (.125) |
| B clear ranked joint | 8 | 40/40 (1.000) | 8/8 | 3/40 (.075) | 26/40 (.650) | 37/40 (.925) | 3/40 (.075) | 39/40 (.975) | 80/120 (.667) | 29/120 (.242) | 0/8 (.000) |
| C clear unranked joint | 8 | 22/40 (.550) | 2/8 | 3/40 (.075) | 15/40 (.375) | 20/40 (.500) | 5/40 (.125) | 22/40 (.550) | 34/120 (.283) | 17/120 (.142) | n/a |
| D single action | 40 | 40/40 (1.000) | 8/8 | 5/40 (.125) | 26/40 (.650) | 23/40 (.575) | 2/40 (.050) | 39/40 (.975) | 80/120 (.667) | 30/120 (.250) | n/a |

Conclusion: clearer formatting fixes B completeness but not spatial judgment; removing ranking does not help and harms completeness; single-action decomposition fixes structure and slightly improves bearing only. Framing remains constant true, collision constant false, and progress is nearly always wrong. The dominant failure is spatial post-state reasoning, with a secondary joint-output burden.
