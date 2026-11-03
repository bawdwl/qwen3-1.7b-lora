"""Validate the bundled original-character dialogue datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_JSONL_ROWS = {
    "datasets/xu-cheng/source/train_sessions.jsonl": 80,
    "datasets/xu-cheng/source/validation_sessions.jsonl": 20,
    "datasets/xu-cheng/source/final_test_sessions.jsonl": 30,
    "datasets/xu-cheng/sft/train.jsonl": 576,
    "datasets/xu-cheng/sft/validation.jsonl": 142,
    "datasets/song-zhiyi/sft/train.jsonl": 276,
    "datasets/song-zhiyi/sft/validation.jsonl": 35,
    "datasets/song-zhiyi/sft/test.jsonl": 35,
}
EXPECTED_JSON = {
    "datasets/song-zhiyi/source/scenarios.json": ("scenarios", 80),
    "datasets/song-zhiyi/source/dialogues.json": ("dialogues", 80),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report: dict[str, dict] = {}
    failures: list[str] = []

    for relative, expected in EXPECTED_JSONL_ROWS.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        rows = read_jsonl(path)
        if len(rows) != expected:
            failures.append(f"{relative}: expected {expected} rows, got {len(rows)}")
        if "/sft/" in f"/{relative}":
            for index, row in enumerate(rows):
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    failures.append(f"{relative}:{index + 1}: missing messages")
                    continue
                if messages[-1].get("role") != "assistant":
                    failures.append(f"{relative}:{index + 1}: last message is not assistant")
                if not str(messages[-1].get("content", "")).strip():
                    failures.append(f"{relative}:{index + 1}: empty assistant target")
        report[relative] = {
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    for relative, (key, expected) in EXPECTED_JSON.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value[key] if isinstance(value, dict) else value
        if not isinstance(rows, list) or len(rows) != expected:
            failures.append(
                f"{relative}: expected {expected} entries under {key}, "
                f"got {len(rows) if isinstance(rows, list) else type(rows).__name__}"
            )
        report[relative] = {
            "rows": len(rows) if isinstance(rows, list) else None,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    output = {
        "status": "failed" if failures else "passed",
        "files": report,
        "failures": failures,
    }
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
