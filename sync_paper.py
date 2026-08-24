#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys

from adaptive_behaviour import paths

HEADER = (
    "% ---------------------------------------------------------------------------\n"
    "% GENERATED FILE - DO NOT EDIT\n"
    "% Produced by the adaptive-behaviour analysis code and delivered here by\n"
    "% sync_paper.py. Edit the analysis and re-run the sync; edits made here are\n"
    "% overwritten, and until then the manuscript disagrees with its own data.\n"
    "% ---------------------------------------------------------------------------\n"
)


def _strip_header(text: str) -> str:
    """Return the table body, so a delivered file compares equal to its source."""
    return text[len(HEADER):] if text.startswith(HEADER) else text


def _pairs() -> list[tuple]:
    """(source, destination, is_text) for everything this repository generates."""
    items = [(t, paths.PAPER_TABLES / t.name, True)
             for t in sorted(paths.OUTPUTS_TABLES.glob("*.tex"))]
    panels = paths.OUTPUTS_FIGURES / "fig_panels.pdf"
    if panels.exists():
        items.append((panels, paths.PAPER_FIGURES / panels.name, False))
    return items


def _differs(src, dst, is_text: bool) -> bool:
    """True when the destination needs rewriting.

    A delivered .tex is out of date if its body differs OR if it is missing the
    do-not-edit header -- otherwise a file that predates the header would never
    acquire one, and the warning would silently never appear.
    """
    if not dst.exists():
        return True
    if is_text:
        current = dst.read_text()
        return not current.startswith(HEADER) or src.read_text() != _strip_header(current)
    return src.read_bytes() != dst.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare without writing; exit 1 if anything differs")
    args = ap.parse_args()

    items = _pairs()
    if not items:
        print(f"nothing to sync: {paths.OUTPUTS_TABLES} is empty", file=sys.stderr)
        return 1

    changed = [(s, d, t) for s, d, t in items if _differs(s, d, t)]

    if args.check:
        for src, dst, _ in changed:
            state = "missing" if not dst.exists() else "differs"
            print(f"  {state:8s} {paths.rel(dst)}")
        if changed:
            print(f"\n{len(changed)} of {len(items)} files out of date -- "
                  f"run: python sync_paper.py")
            return 1
        print(f"all {len(items)} generated files match the paper repository")
        return 0

    paths.PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    for src, dst, is_text in changed:
        if is_text:
            dst.write_text(HEADER + src.read_text())
        else:
            shutil.copyfile(src, dst)
        print(f"  wrote {paths.rel(dst)}")

    if not changed:
        print(f"already up to date ({len(items)} files)")
    else:
        print(f"\n{len(changed)} of {len(items)} files updated in "
              f"{paths.rel(paths.PAPER_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
