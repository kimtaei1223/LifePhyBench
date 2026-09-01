#!/usr/bin/env python3
"""Validate the anonymous TMLR manuscript without requiring a LaTeX runtime."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPOSITORY / "manuscript" / "tmlr"
MAIN = MANUSCRIPT / "main.tex"


def fail(message: str) -> None:
    raise SystemExit(f"TMLR manuscript validation failed: {message}")


def without_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def main() -> None:
    tex_files = [MAIN, *sorted((MANUSCRIPT / "sections").glob("*.tex"))]
    missing_tex = [path for path in tex_files if not path.is_file()]
    if missing_tex:
        fail(f"missing TeX files: {missing_tex}")

    source = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    active_main = without_comments(MAIN.read_text(encoding="utf-8"))

    if "\\usepackage{tmlr}" not in active_main:
        fail("the official review style is not active")
    if re.search(r"\\usepackage\s*\[(?:accepted|preprint)\]\s*\{tmlr\}", active_main):
        fail("accepted/preprint mode must not be enabled for review")
    if "\\author{Anonymous Authors}" not in active_main:
        fail("the source-level author placeholder is not anonymous")

    included = re.findall(r"\\input\{([^}]+)\}", active_main)
    for relative in included:
        target = MANUSCRIPT / relative
        if target.suffix == "":
            target = target.with_suffix(".tex")
        if not target.is_file():
            fail(f"missing included file: {relative}")

    for relative in re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source
    ):
        target = MANUSCRIPT / "figures" / relative
        if not target.is_file():
            fail(f"missing figure: {relative}")

    bibliography = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    cited_keys: set[str] = set()
    for group in re.findall(
        r"\\(?:cite|citep|citet)(?:\[[^]]*\]){0,2}\{([^}]+)\}", source
    ):
        cited_keys.update(key.strip() for key in group.split(","))
    undefined = sorted(cited_keys - bib_keys)
    if undefined:
        fail(f"undefined citation keys: {undefined}")

    private_patterns = {
        "absolute home path": r"/" + r"home/[^/\s{}]+",
        "email address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    }
    for label, pattern in private_patterns.items():
        if re.search(pattern, source):
            fail(f"{label} found in manuscript TeX")

    manifest = (MANUSCRIPT / "STYLE_MANIFEST.txt").read_text(encoding="utf-8")
    entries = re.findall(r"^([0-9a-f]{64})\s+(.+)$", manifest, flags=re.MULTILINE)
    if len(entries) != 3:
        fail("style manifest must contain exactly three hash entries")
    for expected, relative in entries:
        path = MANUSCRIPT / relative
        if not path.is_file():
            fail(f"missing vendored style file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"vendored style changed: {relative}")

    required_sections = {
        "Introduction",
        "Related work",
        "Selective-reset problem setting",
        "Belief supervisor and comparators",
        "Evaluation protocol",
        "Results",
        "Discussion and limitations",
        "Broader impact",
        "Conclusion",
    }
    observed_sections = set(re.findall(r"\\section\{([^}]+)\}", source))
    missing_sections = sorted(required_sections - observed_sections)
    if missing_sections:
        fail(f"missing required sections: {missing_sections}")

    print(
        "TMLR manuscript source validation passed: "
        f"{len(tex_files)} TeX files, {len(cited_keys)} citation keys, "
        f"{len(entries)} pinned style files"
    )


if __name__ == "__main__":
    main()
