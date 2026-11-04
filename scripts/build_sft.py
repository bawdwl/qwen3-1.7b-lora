"""Expand complete multi-turn sessions into assistant-only SFT rows.

Accepted input:

1. JSONL where each line is a session object;
2. JSON containing a list under ``--input-key`` (default: ``dialogues``).

Each session must contain a ``messages`` list that starts with user, alternates
user/assistant, ends with assistant, and carries either ``system_prompt`` or a
system prompt supplied through ``--system-prompt``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--input-key", default="dialogues")
    parser.add_argument(
        "--keep-fields",
        nargs="*",
        default=[
            "session_id",
            "scene_id",
            "storyline_id",
            "category",
            "relationship_stage",
            "support_need",
            "source_session_sha256",
        ],
    )
    return parser.parse_args()


def read_sessions(path: Path, input_key: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get(input_key) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty list of sessions")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every session must be a JSON object")
    return rows


def validate_messages(messages: Any, index: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"session {index}: messages must contain at least one pair")
    if len(messages) % 2:
        raise ValueError(f"session {index}: messages must contain complete pairs")
    cleaned: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        expected = "user" if message_index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(
                f"session {index}: message {message_index} must be role={expected}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"session {index}: message {message_index} has empty content"
            )
        cleaned.append({"role": expected, "content": content.strip()})
    return cleaned


def expand_sessions(
    sessions: Iterable[dict[str, Any]],
    *,
    fallback_system_prompt: str | None,
    keep_fields: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions, 1):
        messages = validate_messages(session.get("messages"), session_index)
        system_prompt = str(
            session.get("system_prompt")
            or session.get("system-prompt")
            or fallback_system_prompt
            or ""
        ).strip()
        if not system_prompt:
            raise ValueError(f"session {session_index}: missing system_prompt")
        session_id = str(session.get("session_id") or session.get("id") or "").strip()
        if not session_id:
            raise ValueError(f"session {session_index}: missing session_id")
        for target_index in range(1, len(messages) // 2 + 1):
            target_message_index = target_index * 2 - 1
            row: dict[str, Any] = {
                "id": f"{session_id}:a{target_index:02d}",
                "target_index": target_index,
                "system_prompt": system_prompt,
                "messages": messages[: target_message_index + 1],
            }
            for field in keep_fields:
                if field in session:
                    row[field] = copy.deepcopy(session[field])
            output.append(row)
    return output


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fallback = (
        args.system_prompt.expanduser().read_text(encoding="utf-8").strip()
        if args.system_prompt
        else None
    )
    sessions = read_sessions(source, args.input_key)
    rows = expand_sessions(
        sessions,
        fallback_system_prompt=fallback,
        keep_fields=list(args.keep_fields),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(output),
                "sessions": len(sessions),
                "assistant_targets": len(rows),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
