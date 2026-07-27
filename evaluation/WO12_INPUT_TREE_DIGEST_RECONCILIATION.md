# WO-11 / WO-12 input-tree digest reconciliation

The WO-11 baseline report records the legacy public-PDF tree digest
`9031e646542f4892a36e07ea42dc34d4a335699a8ced543e1f8038735de40884`.
That historical report does not specify the byte-framing algorithm used to
produce the digest.

The WO-12 v2 grouped-split demonstration records
`21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e`.
Its algorithm is source-defined in `devtools/grouped_split_evidence.py`:

1. Discover exactly 1,000 canonical `.pdf` files.
2. Sort basenames by `(case-insensitive basename, original basename)`.
3. For each file, append the four-byte big-endian length of its UTF-8
   basename, the UTF-8 basename, and the raw 32-byte SHA-256 of the PDF bytes.
4. SHA-256 the resulting stream.

The WO-12 verifier recomputed that digest from bounded snapshots, required the
1,000 case IDs to match the external label-blind layout manifest exactly, and
recomputed every layout signature before emitting aggregate evidence.

Because the WO-11 framing algorithm is not recorded, the two digest strings are
not directly comparable. Their difference is neither evidence of population
drift nor proof of population identity. WO-12 binds its exact observed
population under the source-defined v2 algorithm; future comparisons must use
that same algorithm. No equivalence claim is inferred from the legacy digest.
