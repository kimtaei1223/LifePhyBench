# Clean-checkout reproduction — 2026-08-31

Status: **passed**. Artifact regeneration was first verified at remote commit
`818073068d9b70c2c51ae801ec2c609d1604f27e`; installation from an empty Python
environment was then verified at remote commit
`6d9cd2a68240f32af98bb615d525050c0aabd144`.

## Scope

The private remote `research/lifephybench` branch was cloned into a new empty
directory. No untracked training outputs from the development workspace were
available. The first audit used the verified Python 3.11 project environment.
A second remote clone used a newly created Python 3.11.15 venv, the exact
CPU-only dependency lock, and no inherited `PYTHONPATH`. A download cache was
allowed, but no package from the development environment was inherited.

The following command was executed from the remote clean checkout:

```bash
python scripts/reproduce_clean_checkout.py --run-tests \
  --report clean_checkout_reproduction.json
```

The empty-environment audit used the repository bootstrap command:

```bash
LIFEPHYBENCH_PYTHON=python3.11 \
  ./scripts/bootstrap_reproduction_env.sh .venv-reproduction
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
| Test suite in clean checkout | 154 passed in 16.09 s |
| Exact lock installation in empty Python 3.11 venv | passed |
| `pip check` in empty environment | no broken requirements |
| CPU-only PyTorch model loading | passed with `torch==2.13.0+cpu` |
| Test suite in empty environment | 154 passed in 15.91 s |
| Clean checkout after reproduction | no tracked changes |

The validated dependency lock is
[`requirements-reproduction.txt`](../requirements-reproduction.txt), SHA-256
`cea4f2d5d2cbcd34fc4722985c06f6e59be4348769f13a771a32af9692f5c766`.
It deliberately uses the official CPU-only PyTorch wheel index and installs no
CUDA runtime packages.

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
3. `tensorboard` was present in the development environment but absent from the
   declared project dependencies. Empty-environment test collection exposed
   this omission; it is now declared in the analysis extra and pinned in the
   reproduction lock.

These corrections change no experiment row, statistic, policy selection, or
paper result. They repair publication packaging and cryptographic provenance.

## Remaining reproducibility work

- refresh the dependency lock if the submission date changes substantially;
- repeat this command on the final anonymous submission commit.
