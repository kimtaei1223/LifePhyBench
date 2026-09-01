# Anonymous TMLR manuscript

This directory is the review-version manuscript workspace.

## Template provenance

The `tmlr.sty`, `tmlr.bst`, and bundled `fancyhdr.sty` files were copied from
the official
[`JmlrOrg/tmlr-style-file`](https://github.com/JmlrOrg/tmlr-style-file)
repository at commit:

```text
7bf90efe3a0debbba703c05c43f3ff7e4d4a2992
```

Their Apache-2.0 license is preserved as `LICENSE-TMLR-STYLE`. To satisfy the
repository's no-contact-information rule, two upstream comments containing
third-party email addresses were removed from `tmlr.bst` and `fancyhdr.sty`.
No executable or formatting statement was changed. The resulting files are
bound by `STYLE_MANIFEST.txt`.

## Build

From this directory, use a LaTeX installation with BibTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current workstation did not have `pdflatex` or `latexmk` available when
the skeleton was created. The final review draft has also been compiled with
Tectonic. Independently of the LaTeX engine, source-level citation, figure,
style-hash, evidence, and privacy checks can be run from the repository root:

```bash
python scripts/validate_tmlr_manuscript.py
python scripts/audit_tmlr_claims.py
python scripts/audit_repository_privacy.py
```

## Review anonymity

Keep `\usepackage{tmlr}` without the `accepted` or `preprint` option.
Do not add author names, affiliations, acknowledgments, identified repository
links, machine paths, hostnames, or personal contact information. The source
author field deliberately contains only `Anonymous Authors`, even though the
official review style suppresses it.

## Evidence sources

The manuscript text must remain consistent with:

- `docs/PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`;
- `docs/REACHER_REPLICATION_FINAL_RESULTS.md`;
- `docs/INTEGRATED_PUSHER_REACHER_AUDIT.md`;
- `docs/TMLR_MANUSCRIPT_BLUEPRINT.md`;
- `docs/LITERATURE_AUDIT_2026-08-31.md`.

The current review source contains no unresolved manuscript `TODO` markers.
Any later edit to a principal number should be followed by both manuscript and
claim-to-evidence validation.
