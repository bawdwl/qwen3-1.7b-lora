"""Check relative file links in repository Markdown files."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "codex:", "plugin:", "#")


def strip_fenced_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def main() -> None:
    root = Path.cwd().resolve()
    failures: list[str] = []
    checked = 0
    for markdown in sorted(root.rglob("*.md")):
        if any(part in markdown.parts for part in {".git", ".venv", "outputs"}):
            continue
        text = strip_fenced_code(markdown.read_text(encoding="utf-8"))
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split("#", 1)[0].strip()
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            target = unquote(target.replace("\\", "/"))
            resolved = (markdown.parent / target).resolve()
            checked += 1
            if not resolved.exists():
                failures.append(
                    f"{markdown.relative_to(root)}: missing link target {target}"
                )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"Markdown links: {checked} local targets passed")


if __name__ == "__main__":
    main()
