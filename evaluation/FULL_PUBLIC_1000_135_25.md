# Full-public 1,000-case visible-evidence candidate

This artifact records the aggregate public-training diagnostic for the
answer-key-free visible-evidence finalizer. The 1,000 labels are public and
were used by upstream development, so this is **not** an unseen holdout,
private-test, official-leaderboard, or final-ranking score.

## Candidate boundary

- Parent runtime source: `f928db1a76ce48283d236472b600ca64ae68cef3`
- Stable base predictions:
  `/private/tmp/mib-wo11-full1000-recovered-v2.jsonl`
- Base prediction SHA-256:
  `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`
- Candidate prediction SHA-256:
  `aeece562ef63cc191fde7fee8a0a3e59d231074617be3870daee7871e402c381`
- Aggregate evaluation SHA-256:
  `09e51d7e5d2bafe6d2b8f919958c77c61e8754bda21f8150ba1590f4f0ff33a4`
- Evaluator: `mib_weighted_v1`
- Coverage: `1,000 / 1,000`

The finalizer was first evaluated as an isolated licensed-source adapter and
then from the files integrated into this repository. Both runs produced the
same candidate prediction SHA-256 above. The four integrated runtime files
were also byte-identical to their pinned audited source before repository
integration:

| File | SHA-256 |
| --- | --- |
| `mib_pipeline/score_heads.py` | `ccd64c1211b52ba51950ead4594a7dd259f76326e3f3197802418c905a406068` |
| `mib_pipeline/score_confidence.py` | `8c0a7d60af45637f1c033fdda88542cdf77a4833a17bb3f489586668aeacd183` |
| `mib_pipeline/score_finalizer.py` | `88d37236d32d76559e51b98f0e21f23818b0603b4075cfb6ae62520e77e15fa9` |
| `mib_pipeline/artifacts/score_confidence_blend.json` | `129feb88eb08b2583a5ed88a9cee5fa9965f7bb0d68480783d822796389bcff6` |

## Aggregate result

| Component | Governed baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Extraction | `44.87777777777778` | `44.97888888888889` | `+0.10111111111111` |
| Classification | `68.52000000000001` | `72.69` | `+4.17` |
| Calibration | `16.974076455665433` | `17.58246401876988` | `+0.608387563104447` |
| **Total** | **`130.37185423344323`** | **`135.25135290765877`** | **`+4.87949867421554`** |

| Quality measure | Candidate |
| --- | ---: |
| Missing / extra / duplicate cases | `0 / 0 / 0` |
| Invalid adjudication / confidence / fee rows | `0 / 0 / 0` |
| Catastrophic false approvals | `0` |
| Mean confidence Brier error | `0.06043839953075305` |

The committed aggregate JSON is
`evaluation/FULL_PUBLIC_1000_135_25.json`. No prediction rows, public answers,
case-level score tables, validation outputs, or case-identity lookups are
committed.

## Verification

- Focused integration tests: `26 passed`.
- Full host suite: `393 passed, 2 skipped`.
- Ten-case production-entry-point smoke test:
  `attempted=10 answered=10 omitted=0`.
- The ten smoke-test rows were byte-identical to the corresponding rows in the
  independently finalized 1,000-case candidate.
- Runtime scan found no `MIB_ALLOW_ANSWER_KEY`,
  `apply_answer_key_transcription`, answer-key module, or six-digit runtime
  case literal.

The exact 1,000-row production-base OCR run is reused rather than repeated:
only the newly added deterministic outer layer changed. A clean container
build and constrained 5,000-case validation run remain separate runtime and
submission gates.

## Source and safety decision

The layer is derived from public MIT-licensed challenge code with exact
attribution in `ATTRIBUTION.md`. Two nominally higher public-training variants
were rejected because their documented default paths consume embedded fake
answer-key text; the highest also adds a literal public-label-derived approval
allowlist. Neither unsafe path exists in this candidate.
