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
    "datasets/pei-jiming/sft/train.jsonl": 577,
    "datasets/pei-jiming/sft/validation.jsonl": 54,
    "datasets/pei-jiming/sft/test.jsonl": 45,
}
EXPECTED_JSON = {
    "datasets/song-zhiyi/source/scenarios.json": ("scenarios", 80),
    "datasets/song-zhiyi/source/dialogues.json": ("dialogues", 80),
    "datasets/pei-jiming/source/scenarios.json": ("scenarios", 120),
    "datasets/pei-jiming/source/dialogues.json": ("dialogues", 120),
    "datasets/pei-jiming/source/long_reply_supplement.json": ("samples", 60),
}
DATASET_INFO = {
    "datasets/xu-cheng/sft/dataset_info.json": {
        "xu_cheng_train": "train.jsonl",
        "xu_cheng_validation": "validation.jsonl",
    },
    "datasets/song-zhiyi/sft/dataset_info.json": {
        "song_zhiyi_train": "train.jsonl",
        "song_zhiyi_validation": "validation.jsonl",
        "song_zhiyi_test": "test.jsonl",
    },
    "datasets/pei-jiming/sft/dataset_info.json": {
        "pei_jiming_train": "train.jsonl",
        "pei_jiming_validation": "validation.jsonl",
        "pei_jiming_test": "test.jsonl",
    },
}
CHARACTER_SLUGS = {
    "宋知意": "song-zhiyi",
    "许澄": "xu-cheng",
    "裴既明": "pei-jiming",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def character_slug(character: str) -> str:
    try:
        return CHARACTER_SLUGS[character]
    except KeyError as exc:
        raise ValueError(f"Unknown character in weights manifest: {character}") from exc


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
    session_splits: dict[tuple[str, str], set[str]] = {}

    for relative, expected in EXPECTED_JSONL_ROWS.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        rows = read_jsonl(path)
        if len(rows) != expected:
            failures.append(f"{relative}: expected {expected} rows, got {len(rows)}")
        if "/sft/" in f"/{relative}":
            target_lengths: list[int] = []
            session_ids: set[str] = set()
            for index, row in enumerate(rows):
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    failures.append(f"{relative}:{index + 1}: missing messages")
                    continue
                if len(messages) % 2:
                    failures.append(f"{relative}:{index + 1}: odd message count")
                for message_index, message in enumerate(messages):
                    expected_role = "user" if message_index % 2 == 0 else "assistant"
                    if message.get("role") != expected_role:
                        failures.append(
                            f"{relative}:{index + 1}: message {message_index} "
                            f"expected role={expected_role}"
                        )
                    if not str(message.get("content", "")).strip():
                        failures.append(
                            f"{relative}:{index + 1}: empty message {message_index}"
                        )
                if messages[-1].get("role") != "assistant":
                    failures.append(f"{relative}:{index + 1}: last message is not assistant")
                if not str(messages[-1].get("content", "")).strip():
                    failures.append(f"{relative}:{index + 1}: empty assistant target")
                else:
                    target_lengths.append(len(messages[-1]["content"].strip()))
                if not str(row.get("system_prompt", "")).strip():
                    failures.append(f"{relative}:{index + 1}: missing system_prompt")
                session_id = str(row.get("session_id", "")).strip()
                if not session_id:
                    failures.append(f"{relative}:{index + 1}: missing session_id")
                else:
                    session_ids.add(session_id)
            company = relative.split("/")[1]
            split = path.stem
            session_splits[(company, split)] = session_ids
            target_lengths.sort()
            if target_lengths:
                p50 = target_lengths[(len(target_lengths) - 1) // 2]
                p90 = target_lengths[round((len(target_lengths) - 1) * 0.9)]
            else:
                p50 = p90 = 0
            report[f"{relative}#target_chars"] = {
                "min": min(target_lengths) if target_lengths else 0,
                "p50": p50,
                "p90": p90,
                "max": max(target_lengths) if target_lengths else 0,
            }
        report[relative] = {
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    for company, splits in {
        "song-zhiyi": ("train", "validation", "test"),
        "pei-jiming": ("train", "validation", "test"),
    }.items():
        seen: dict[str, str] = {}
        for split in splits:
            for session_id in session_splits.get((company, split), set()):
                if session_id in seen and seen[session_id] != split:
                    failures.append(
                        f"{company}: session {session_id} leaks from "
                        f"{seen[session_id]} to {split}"
                    )
                seen[session_id] = split

    for relative, expected in DATASET_INFO.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{relative}: invalid JSON: {exc}")
            continue
        for name, expected_file in expected.items():
            if value.get(name, {}).get("file_name") != expected_file:
                failures.append(
                    f"{relative}: {name} must register file_name={expected_file}"
                )
        report[relative] = {
            "entries": sorted(value),
            "sha256": sha256(path),
        }

    manifest_path = root / "datasets/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for list_name in ("xu_cheng", "song_zhiyi", "pei_jiming"):
            item = manifest.get(list_name, {})
            for relative, metadata in item.get("files", {}).items():
                path = root / relative
                if not path.is_file():
                    failures.append(f"manifest: missing {relative}")
                    continue
                actual_hash = sha256(path)
                if actual_hash.lower() != str(metadata.get("sha256", "")).lower():
                    failures.append(f"manifest: SHA-256 mismatch for {relative}")
                expected_rows = metadata.get("rows")
                if expected_rows is not None and relative.endswith(".jsonl"):
                    rows = read_jsonl(path)
                    if len(rows) != expected_rows:
                        failures.append(
                            f"manifest: row mismatch for {relative}: "
                            f"expected {expected_rows}, got {len(rows)}"
                        )
    else:
        failures.append("missing: datasets/manifest.json")

    weights_manifest_path = root / "weights/manifest.json"
    if weights_manifest_path.is_file():
        weights_manifest = json.loads(
            weights_manifest_path.read_text(encoding="utf-8")
        )
        for adapter in weights_manifest.get("adapters", []):
            character = adapter.get("character", "unknown")
            checkpoint = adapter.get("checkpoint", "unknown")
            adapter_dir = root / "weights" / f"{character_slug(character)}-{checkpoint}"
            for file_name, metadata in adapter.get("files", {}).items():
                path = adapter_dir / file_name
                if not path.is_file():
                    failures.append(
                        f"weights manifest: missing {path.relative_to(root)}"
                    )
                    continue
                actual_hash = sha256(path)
                if actual_hash.lower() != str(metadata.get("sha256", "")).lower():
                    failures.append(
                        f"weights manifest: SHA-256 mismatch for "
                        f"{path.relative_to(root)}"
                    )
                expected_bytes = metadata.get("bytes")
                if expected_bytes is not None and path.stat().st_size != expected_bytes:
                    failures.append(
                        f"weights manifest: byte-size mismatch for "
                        f"{path.relative_to(root)}"
                    )
        report["weights/manifest.json"] = {
            "adapters": len(weights_manifest.get("adapters", [])),
            "sha256": sha256(weights_manifest_path),
        }
    else:
        failures.append("missing: weights/manifest.json")

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
