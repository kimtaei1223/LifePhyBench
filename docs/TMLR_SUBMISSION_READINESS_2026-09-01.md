# TMLR submission readiness — 2026-09-01

## Verdict

The anonymous manuscript and reproducibility supplement are **machine-ready**.
They are not yet authorized for upload because the remaining gates require
author decisions or OpenReview account state that cannot be inferred from the
repository.

This checkpoint follows the current official TMLR
[editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html),
[author guide](https://www.jmlr.org/tmlr/author-guide.html), and
[acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html).

## Built review artifacts

| Artifact | Size | SHA-256 | Result |
|---|---:|---|---|
| `TMLR_REVIEW_MANUSCRIPT.pdf` | 320,666 bytes; 9 pages | `b04c071c8291c70c8377403440db7b2571d184308fb0f3873bb6550c4eb5ebbd` | PASS |
| `TMLR_SUPPLEMENT_ANONYMOUS.zip` | 18,616,087 bytes; 331 files | `27a61a7653bc18e2dce428c04004777180bc4410284b661ef8c0d6606ebfeeac` | PASS; below 100 MB |

The local build outputs are stored under ignored directory `submission/`.
`SUBMISSION_MANIFEST.json` records both hashes.

## Automated gates

| Gate | Result |
|---|---|
| Official review style active; review author placeholder anonymous | PASS |
| Source citations, figures, style hashes, and evidence sentinels | PASS; 17 citation keys |
| Claim-to-evidence trace | PASS; 38 numerical and interpretation checks |
| Repository, PDF, and ZIP privacy scan | PASS |
| Complete unit/semantic test suite | PASS; 154 tests |
| PDF generation | PASS; 9 letter-size pages, no undefined citations or overfull boxes |
| Supplement size and format | PASS; ZIP, 17.8 MiB versus 100 MiB limit |
| Supplement reproducibility from extraction without Git metadata | PASS |
| Sealed manifest/protocol hashes | PASS; 102 artifacts and 4 protocols |
| Retained model loading on CPU | PASS; 2 archives |
| Publication-artifact regeneration | PASS; 11 Pusher and 7 Reacher artifacts |
| Deterministic supplement build | PASS; repeated build produced identical SHA-256 |

The final Tectonic build emits no layout, citation, or reference warnings.
Visual inspection found no clipped content, broken table, unresolved reference,
or identity leak.

## Final scientific framing

The paper is an empirical two-task study, not a new safety-filter algorithm.
Its defensible lesson is that persistent hidden state should be evaluated over
complete lifetimes, and that mean utility, an empirical point-rate gate, and
calibration evidence are different claims. The manuscript explicitly preserves
the following negative or boundary evidence:

- the residual's incremental Pusher effect was not established;
- both original physics-only policies missed the 2% point rule;
- the later inherited Reacher policy also passed on the extension sample;
- calibration necessity and superiority were not established;
- only 52/100 selected-policy lifetime effects were positive;
- the thermal law has no hardware or physical-unit validation.

## Human-only gates before upload

- Freeze the exact author list; obtain every author's approval and confirm that
  each person satisfies TMLR's authorship criteria.
- Confirm that every author has a complete active OpenReview profile, correct
  three-year institutional history, personal conflicts, and publication record.
- Confirm every author's remaining annual TMLR submission quota.
- Confirm that no text, figure, or result is published, accepted, or under
  parallel review at another archival peer-reviewed venue.
- Enter funding, competing interests, human-subject/IRB disposition, and any
  additional conflicts in OpenReview.
- Select and recommend an appropriate action editor.
- Accept that the submission is licensed under CC BY 4.0 from submission onward.
- Repeat the targeted Level-1 literature search and official-policy check if the
  actual upload date is later than 2026-09-01.

## Rebuild command

After compiling `main.tex`, build and re-audit both upload artifacts with:

```bash
python scripts/build_tmlr_submission.py \
  --pdf /path/to/main.pdf \
  --output-dir submission
```

Any manuscript change after this checkpoint invalidates the recorded PDF hash
and requires rerunning the claim audit, tests, PDF build, and package builder.
