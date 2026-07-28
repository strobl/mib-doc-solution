# Attribution and provenance

This solution combines and modifies public MIT-licensed challenge entries.
Validation predictions are generated locally by this repository; no other
participant's validation output is copied.

## Organizer materials

- Repository: <https://github.com/8090-inc/mib-doc-challenge>
- License: MIT

The schemas, evaluator, Docker runner, field manual, public labels, and
synthetic dataset are organizer materials.

## Visible-evidence final scoring layer

- Audited source:
  <https://github.com/vibemarketer94/mib-doc-solution>
- Pinned source commit:
  `499ba035846134b2293fa1e9bbf58c870ba51513`
- Original visible-head source:
  <https://github.com/arjunkshah12345-hash/mib-doc-solution>
- Pinned original commit:
  `798ad2277a89a25cd3dfc596be572df4aade55c6`
- License: MIT; the original notice is retained at
  `third_party_licenses/PublicSolutions/LICENSE-Arjun`

The adapted layer includes visible field repairs, layout consensus, narrow
visible denial signals, finding/damage handling, approval safety demotions,
denial softening, an identity-free confidence table, and a two-parameter
confidence transform.

The runtime intentionally excludes all embedded answer-key transcription,
case-ID lookup, filename lookup, participant validation predictions, and broad
public-label allowlists. Embedded generator instructions remain untrusted.
The batch boundary fails closed to the valid base prediction if this optional
layout-only layer raises an exception.

## OCR models and dependencies

RapidOCR/PaddleOCR model provenance and Apache-2.0 notices are retained under
`third_party_licenses/`. Installed Python wheels keep their upstream license
and notice files inside the image. See
`third_party_licenses/MODEL_PROVENANCE.md` and
`third_party_licenses/README.md`.
