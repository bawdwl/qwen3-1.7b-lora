"""Convert Chat-Huanhuan JSON to LLaMA-Factory ShareGPT messages."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


DATA_VERSION = "huanhuan-author-close-v1"
SOURCE_URL = (
    "https://github.com/KMnO4-zx/huanhuan-chat/"
    "blob/master/dataset/train/lora/huanhuan.json"
)
DEFAULT_SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def summary(values: list[int]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
    }


def convert_rows(source_rows: list[dict], system_prompt: str) -> tuple[list[dict], dict]:
    converted: list[dict] = []
    pair_counter: Counter[tuple[str, str, str]] = Counter()
    for source_index, source in enumerate(source_rows, 1):
        missing = {"instruction", "input", "output"} - set(source)
        if missing:
            raise ValueError(f"row {source_index} missing fields: {sorted(missing)}")
        instruction = str(source["instruction"] or "")
        additional_input = str(source["input"] or "")
        output = str(source["output"] or "")
        if not instruction.strip():
            raise ValueError(f"row {source_index} has an empty instruction")
        if not output.strip():
            raise ValueError(f"row {source_index} has an empty output")
        pair_counter[(instruction, additional_input, output)] += 1
        converted.append(
            {
                "id": f"huanhuan-official-{source_index:05d}",
                "source_row": source_index,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction + additional_input},
                    {"role": "assistant", "content": output},
                ],
            }
        )
    audit = {
        "source_rows": len(source_rows),
        "converted_rows": len(converted),
        "exact_duplicate_pairs_retained": sum(
            count - 1 for count in pair_counter.values()
        ),
        "unique_instruction": len(
            {str(row["instruction"]) for row in source_rows}
        ),
        "unique_output": len({str(row["output"]) for row in source_rows}),
        "character_lengths": {
            "question": summary(
                [
                    len(str(row["instruction"]) + str(row["input"] or ""))
                    for row in source_rows
                ]
            ),
            "answer": summary(
                [len(str(row["output"])) for row in source_rows]
            ),
        },
    }
    return converted, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("The source file must contain a JSON array")
    rows, audit = convert_rows(source_rows, args.system_prompt)
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "train.jsonl", rows)
    (output_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "huanhuan_author_close": {
                    "file_name": "train.jsonl",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                        "system_tag": "system",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": DATA_VERSION,
        "source_url": SOURCE_URL,
        "source_sha256": sha256_file(source_path),
        "system_prompt": args.system_prompt,
        "system_prompt_sha256": sha256_text(args.system_prompt),
        "split": {"train": len(rows), "validation": 0, "test": 0},
        "audit": audit,
        "notice": (
            "The source is derived from a screenplay and has non-commercial "
            "usage language upstream. Review data rights before redistribution."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
