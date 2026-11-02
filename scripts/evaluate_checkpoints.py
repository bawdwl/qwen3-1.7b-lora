"""Compare Qwen3 base and LoRA checkpoints on fixed chat probes."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_chat import ChatEngine  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SINGLE_PROMPTS = [
    ("greeting", "你好。"),
    ("identity", "你是谁？"),
    ("tired", "今天朕有些累，不想批折子了。"),
    ("food", "御膳房送来的点心太甜，朕不喜欢。"),
    ("court", "后宫这些人总在争宠，你怎么看？"),
    ("affection", "如果朕今晚只陪你说会儿话，好不好？"),
    ("criticism", "你为什么总是重复同一句话？"),
    ("travel", "朕想离宫去民间住几天，你愿意一起去吗？"),
    ("boundary", "有人传言你干涉朝政，你怎么回应？"),
    ("ending", "今天就算了，改日再聊。"),
    ("modern", "你知道人工智能是什么吗？"),
    ("relationship", "如果朕不是皇帝，只是一个普通人，你还会陪朕吗？"),
]
MULTI_TURN_PROMPTS = [
    ("greeting", "你好。"),
    ("stress", "今天朝里事情很多，朕有些烦。"),
    ("request", "你愿意陪朕说会儿话吗？"),
    ("fatigue", "朕有点累，不想听劝。"),
    ("criticism", "你为什么总说同样的话？"),
    ("topic_shift", "今天我们不说宫里的事。"),
    ("inner_thought", "给朕讲讲你刚才在想什么。"),
    ("ending", "好了，今天就到这里。"),
]
ROLE_MARKERS = ("臣妾", "嫔妾", "妾身", "嬛儿", "皇上", "陛下", "万福")
AI_MARKERS = ("作为AI", "作为 AI", "语言模型", "人工智能助手", "你的助手")


def normalize(text: str) -> str:
    return "".join(
        char
        for char in text.casefold()
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def similarity(left: str, right: str) -> float:
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def repeated_phrase(text: str) -> bool:
    text = normalize(text)
    for size in (4, 5, 6):
        grams = [
            text[index : index + size]
            for index in range(max(0, len(text) - size + 1))
        ]
        if any(count >= 2 for count in Counter(grams).values()):
            return True
    return False


def adapter_modes(adapter_root: Path) -> dict[str, Path]:
    modes: dict[str, Path] = {}
    for path in sorted(adapter_root.glob("checkpoint-*")):
        if (path / "adapter_config.json").is_file():
            modes[path.name.replace("checkpoint-", "step")] = path
    if (adapter_root / "adapter_config.json").is_file():
        modes["final"] = adapter_root
    if not modes:
        raise FileNotFoundError(f"No checkpoint-* adapters under {adapter_root}")
    return modes


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    modes = {"base": Path(".")}
    modes.update(adapter_modes(args.adapter_root))
    engine = ChatEngine(
        args.base_model,
        {name: path for name, path in modes.items() if name != "base"},
        input_budget=args.input_budget,
    )
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8").strip()
        if args.system_prompt
        else ""
    )

    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for index, mode in enumerate(modes, 1):
        print(f"[{index}/{len(modes)}] {mode}", flush=True)
        single: list[dict] = []
        for prompt_id, prompt in SINGLE_PROMPTS:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            answer, metrics = engine.reply(
                messages,
                mode=mode,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                "model": mode,
                "suite": "single",
                "turn": len(single) + 1,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "answer": answer,
                "characters": len(answer),
                "too_short": len(answer) < 4,
                "role_marker": any(marker in answer for marker in ROLE_MARKERS),
                "assistant_voice": any(marker in answer for marker in AI_MARKERS),
                "repeated_phrase": repeated_phrase(answer),
                "question_echo": similarity(answer, prompt) >= 0.7,
                **metrics,
            }
            rows.append(row)
            single.append(row)

        multi: list[dict] = []
        history: list[dict[str, str]] = []
        for turn, (prompt_id, prompt) in enumerate(MULTI_TURN_PROMPTS, 1):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history)
            messages.append({"role": "user", "content": prompt})
            answer, metrics = engine.reply(
                messages,
                mode=mode,
                max_new_tokens=args.max_new_tokens,
            )
            recent_similarity = [
                round(similarity(answer, item["answer"]), 4)
                for item in multi[-3:]
            ]
            row = {
                "model": mode,
                "suite": "multi",
                "turn": turn,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "answer": answer,
                "characters": len(answer),
                "too_short": len(answer) < 4,
                "role_marker": any(marker in answer for marker in ROLE_MARKERS),
                "assistant_voice": any(marker in answer for marker in AI_MARKERS),
                "repeated_phrase": repeated_phrase(answer),
                "question_echo": False,
                "recent_answer_similarity": recent_similarity,
                "near_repeat_recent": bool(
                    recent_similarity and max(recent_similarity) >= 0.75
                ),
                **metrics,
            }
            rows.append(row)
            multi.append(row)
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )

        summaries[mode] = {
            "single_exact_duplicate_answers": sum(
                count - 1
                for count in Counter(
                    normalize(row["answer"]) for row in single
                ).values()
            ),
            "single_role_marker_rate": round(
                sum(row["role_marker"] for row in single) / len(single), 3
            ),
            "single_assistant_voice_rate": round(
                sum(row["assistant_voice"] for row in single) / len(single), 3
            ),
            "single_too_short_rate": round(
                sum(row["too_short"] for row in single) / len(single), 3
            ),
            "single_repeated_phrase_rate": round(
                sum(row["repeated_phrase"] for row in single) / len(single), 3
            ),
            "multi_near_repeat_rate": round(
                sum(row["near_repeat_recent"] for row in multi) / len(multi),
                3,
            ),
            "median_characters": statistics.median(
                row["characters"] for row in [*single, *multi]
            ),
            "median_seconds": statistics.median(
                row["seconds"] for row in single
            ),
        }

    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "base_model": str(args.base_model),
        "adapter_root": str(args.adapter_root),
        "system_prompt_file": str(args.system_prompt) if args.system_prompt else None,
        "system_prompt": system_prompt,
        "single_prompts": len(SINGLE_PROMPTS),
        "multi_turn_prompts": len(MULTI_TURN_PROMPTS),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Checkpoint comparison",
        "",
        f"System prompt: `{system_prompt}`",
        "",
        "| model | duplicate singles | role marker | assistant voice | too short | repeated phrase | multi near-repeat | median chars | median sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, stats in summaries.items():
        lines.append(
            f"| {mode} | {stats['single_exact_duplicate_answers']} | "
            f"{stats['single_role_marker_rate']:.1%} | "
            f"{stats['single_assistant_voice_rate']:.1%} | "
            f"{stats['single_too_short_rate']:.1%} | "
            f"{stats['single_repeated_phrase_rate']:.1%} | "
            f"{stats['multi_near_repeat_rate']:.1%} | "
            f"{stats['median_characters']} | {stats['median_seconds']:.2f} |"
        )
    lines.extend(["", "## Raw answers", ""])
    for mode in modes:
        lines.append(f"### {mode}")
        for row in [r for r in rows if r["model"] == mode]:
            suite = "single" if row["suite"] == "single" else f"multi-{row['turn']}"
            lines.append(f"- **{suite} / {row['prompt_id']}** User: {row['prompt']}")
            lines.append(f"  - Model: {row['answer']}")
        lines.append("")
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--input-budget", type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens must be between 1 and 1024")
    if not 1024 <= args.input_budget <= 32768:
        parser.error("--input-budget must be between 1024 and 32768")
    run(args)


if __name__ == "__main__":
    main()
