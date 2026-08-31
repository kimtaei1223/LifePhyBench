# Clean-checkout reproduction — 2026-08-31

Status: **passed** for remote commit
`818073068d9b70c2c51ae801ec2c609d1604f27e`.

## Scope

The private remote `research/lifephybench` branch was cloned into a new empty
directory. No untracked training outputs from the development workspace were
available. The audit used the verified Python 3.11 dependency environment whose
package versions are recorded in the sealed snapshot; dependency installation
from an empty package cache was not part of this checkpoint.

The following command was executed from the remote clean checkout:

```bash
python scripts/reproduce_clean_checkout.py --run-tests \
  --report clean_checkout_reproduction.json
```

## Verified results

| Check | Result |
|---|---:|
| Repository privacy audit | passed |
| Integrated snapshot root entries | 6/6 |
| Integrated artifact entries | 102/102 |
| Frozen or privacy-redacted protocol bindings | 4/4 |
| Stable-Baselines model archives loaded on CPU | 2/2 |
| Regenerated Pusher publication artifacts | 11/11 hash-identical |
| Regenerated Reacher publication artifacts | 7/7 hash-identical |
| Test suite | 154 passed in 16.09 s |
| Clean checkout after reproduction | no tracked changes |

The tracked-tree review archive was also generated with `git archive`. Its size
was 19,092,310 bytes and its SHA-256 digest was
`fca7420dbd13b4227ae9a6417875cc5faffdccb3287c81139ad15abf66ac0dcf`.
The applicable upload limit must still be checked again at submission time.

## Defects found and corrected during this audit

1. The integrated artifact manifest referenced two runtime `campaign.log`
   files excluded by Git. Snapshot creation now removes runtime logs before
   hashing, and the final manifest contains only tracked files.
2. Publication privacy redaction changed protocol-file hashes while result
   files correctly retained their original frozen-protocol hashes. Explicit
   redaction ledgers now bind each original protocol hash to its sanitized file
   hash and state that result values were unchanged.

These corrections change no experiment row, statistic, policy selection, or
paper result. They repair publication packaging and cryptographic provenance.

## Remaining reproducibility work

- test installation from an empty dependency environment rather than reuse the
  verified project environment;
- refresh the dependency lock if the submission date changes substantially;
- repeat this command on the final anonymous submission commit.
