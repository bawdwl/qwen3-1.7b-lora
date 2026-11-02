"""Interactive terminal chat for a Qwen3 model and optional PEFT adapter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_chat import ChatEngine  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--input-budget", type=int, default=8192)
    args = parser.parse_args()

    if not 1 <= args.max_new_tokens <= 1024:
        parser.error("--max-new-tokens must be between 1 and 1024")
    if not 1024 <= args.input_budget <= 32768:
        parser.error("--input-budget must be between 1024 and 32768")
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8").strip()
        if args.system_prompt
        else ""
    )
    adapters = {"lora": args.adapter} if args.adapter else None
    engine = ChatEngine(
        args.base_model,
        adapters,
        input_budget=args.input_budget,
    )
    mode = "lora" if args.adapter else "base"
    print(f"Model loaded. adapter={bool(args.adapter)} system_prompt={bool(system_prompt)}")
    print("Use /reset to clear the chat and /exit to quit.\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"/exit", "exit", "quit"}:
            break
        if text.lower() in {"/reset", "reset", "clear", "/clear"}:
            history.clear()
            print("Chat cleared.\n")
            continue
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        answer, metrics = engine.reply(
            messages,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        print(f"\nModel> {answer}")
        print(
            f"[{mode} | input {metrics['input_tokens']} tokens"
            f" | output {metrics['output_tokens']} tokens"
            f" | {metrics['seconds']:.2f}s]\n"
        )


if __name__ == "__main__":
    main()
