# WO17 Final-Review Ordinary-Policy Replay — Result

Status: **rejected; no effect; not promoted**.

The exact clean causal source
`dba28d7283ae333265d0f22e99900f279748d69c` produced all 1,000 expected
records in 2,631.11 seconds. Its three runtime-variable files are byte-identical
to integrated source `9583b610c284c82e3f7932c5449cad28b8502f7b`.

The candidate artifact SHA-256 is
`d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`,
which is exactly the frozen baseline artifact SHA-256. Consequently:

- total score remains `130.37185423344323`;
- extraction, classification, and calibration deltas are all `0`;
- all 1,000 rows, all nine extraction fields, decisions, and confidence values
  are unchanged;
- the strict improvement, repeat-positive, and leave-one-fold-out-positive
  gates fail;
- safety, completeness, identity, immutability, and confidence-supplement
  checks pass.

The experiment is therefore recorded as `reject` with rationale
`no_effect_local_diagnostic`. No candidate state is promoted. Trusted Docker,
resource-envelope, and deterministic-repeat release gates were not run because
the local score gate already failed.

The macOS timing wrapper returned nonzero after the solution completed because
its final `kern.clockrate` query was sandbox-denied. This did not affect the
solution result: the solution reported `attempted=1000 answered=1000 omitted=0`
and finalized a valid 1,000-row artifact.
