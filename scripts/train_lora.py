"""Reproducible Qwen3 single-character LoRA training.

The training sample format is one JSON object per line:

{
  "system_prompt": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}

Only the final assistant answer and its EOS token are supervised. System and
history tokens are masked with -100. If an input is too long, the oldest
complete user/assistant turns are removed before the final user turn and
target answer. The script never silently truncates the supervised target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IGNORE_INDEX = -100
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", help="Override model_name_or_path")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize and audit every row without loading the language model",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Limit rows for a connectivity/debug smoke test",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def validate_row(row: dict[str, Any], source: str, index: int) -> None:
    system_prompt = row.get("system_prompt")
    messages = row.get("messages")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(f"{source}:{index}: missing system_prompt")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{source}:{index}: messages must contain at least one pair")
    if len(messages) % 2:
        raise ValueError(f"{source}:{index}: messages must contain user/assistant pairs")
    for message_index, message in enumerate(messages):
        expected = "user" if message_index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(
                f"{source}:{index}: message {message_index} must be role={expected}"
            )
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(
                f"{source}:{index}: message {message_index} has empty content"
            )


def dataset_path(config: dict[str, Any], dataset_name: str) -> Path:
    dataset_dir = Path(config["dataset_dir"]).expanduser()
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if dataset_name not in info:
        raise KeyError(
            f"{dataset_name!r} is absent from {info_path}; "
            f"available: {', '.join(sorted(info))}"
        )
    file_name = info[dataset_name].get("file_name")
    if not file_name:
        raise ValueError(f"{info_path}: {dataset_name} has no file_name")
    return dataset_dir / file_name


def chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_example(
    tokenizer,
    row: dict[str, Any],
    cutoff_len: int,
    *,
    long_target_weight: float,
    long_target_char_threshold: int,
) -> dict[str, Any]:
    system = row["system_prompt"].strip()
    messages = list(row["messages"])
    target = messages[-1]["content"].strip()
    context = messages[:-1]
    eos = tokenizer.eos_token or ""
    if not eos:
        raise ValueError("Tokenizer has no EOS token")

    while True:
        rendered_messages = [{"role": "system", "content": system}, *context]
        prefix = chat_template(tokenizer, rendered_messages)
        full_text = prefix + target + eos
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Tokenization is not prefix-stable; refusing to create an "
                "incorrect assistant-only label mask"
            )
        if len(full_ids) <= cutoff_len:
            break
        if len(context) <= 1:
            raise ValueError(
                f"System prompt, final user turn and answer exceed cutoff_len="
                f"{cutoff_len}; increase the cutoff or shorten the prompt/target"
            )
        # Preserve the final user/assistant pair and remove the oldest complete
        # history pair first.
        context = context[2:]

    labels = [IGNORE_INDEX] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    supervised = [token for token in labels if token != IGNORE_INDEX]
    if not supervised:
        raise ValueError("No supervised target tokens")
    raw_weight = (
        long_target_weight
        if long_target_char_threshold > 0
        and len(target) >= long_target_char_threshold
        else 1.0
    )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sample_weight": float(raw_weight),
        "target_characters": len(target),
        "input_tokens": len(full_ids),
        "supervised_tokens": len(supervised),
        "trimmed_history_pairs": (len(messages) - 1 - len(context)) // 2,
    }


def token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        position = round((len(ordered) - 1) * fraction)
        return ordered[position]

    input_tokens = [row["input_tokens"] for row in rows]
    supervised = [row["supervised_tokens"] for row in rows]
    return {
        "rows": len(rows),
        "input_tokens": {
            "min": min(input_tokens),
            "p50": percentile(input_tokens, 0.5),
            "p90": percentile(input_tokens, 0.9),
            "max": max(input_tokens),
        },
        "supervised_tokens": {
            "min": min(supervised),
            "p50": percentile(supervised, 0.5),
            "p90": percentile(supervised, 0.9),
            "max": max(supervised),
            "total": sum(supervised),
        },
        "trimmed_history_pairs": sum(row["trimmed_history_pairs"] for row in rows),
    }


def load_encoded_dataset(
    tokenizer,
    path: Path,
    cutoff_len: int,
    *,
    long_target_weight: float,
    long_target_char_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = read_jsonl(path)
    encoded: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, 1):
        validate_row(row, str(path), index)
        encoded.append(
            encode_example(
                tokenizer,
                row,
                cutoff_len,
                long_target_weight=long_target_weight,
                long_target_char_threshold=long_target_char_threshold,
            )
        )
    return raw_rows, encoded


def load_tokenizer(base_model: str, revision: str | None):
    from transformers import AutoTokenizer

    local_only = Path(base_model).exists()
    return AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=False,
        local_files_only=local_only,
    )


def build_report(
    config_path: Path,
    config: dict[str, Any],
    base_model: str,
    tokenizer,
    train_path: Path,
    eval_path: Path | None,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    system_prompts = {row["system_prompt"] for row in read_jsonl(train_path)}
    report: dict[str, Any] = {
        "schema_version": "shiyu-lora-training-audit-v1",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_model": base_model,
        "base_revision": config.get("model_revision"),
        "tokenizer_class": tokenizer.__class__.__name__,
        "dataset_dir": config["dataset_dir"],
        "train_dataset": config["dataset"],
        "eval_dataset": config.get("eval_dataset"),
        "train_file": str(train_path),
        "train_file_sha256": sha256_file(train_path),
        "eval_file": str(eval_path) if eval_path else None,
        "eval_file_sha256": sha256_file(eval_path) if eval_path else None,
        "system_prompt_sha256": sorted(sha256_text(value) for value in system_prompts),
        "train": token_stats(train_rows),
        "validation": token_stats(eval_rows) if eval_rows else None,
        "label_policy": {
            "supervised": "final assistant response and EOS",
            "masked": "system, user and prior assistant/history tokens",
            "ignore_index": IGNORE_INDEX,
            "silent_target_truncation": False,
        },
        "sample_weighting": {
            "threshold_characters": config.get("long_target_char_threshold"),
            "long_target_weight": config.get("long_target_weight"),
            "normalization": "mean sample weight per optimizer micro-batch",
        },
    }
    return report


def run_dry(
    report: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = train_rows if eval_rows is None else [*train_rows, *eval_rows]
    if any(row["supervised_tokens"] <= 0 for row in required):
        raise ValueError("A row has no supervised tokens")
    if any(
        any(label != IGNORE_INDEX for label in row["labels"][: row["input_tokens"] - row["supervised_tokens"]])
        for row in required[:1]
    ):
        raise ValueError("History transcript leaked into supervised labels")


def train(config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the tested BF16 recipe. "
            "Install the CUDA build of PyTorch and run nvidia-smi."
        )

    base_model = args.base_model or config["model_name_or_path"]
    revision = config.get("model_revision")
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new path or remove the old experiment after preserving it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(base_model, revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cutoff_len = int(config.get("cutoff_len", 1024))
    long_weight = float(config.get("long_target_weight", 1.0))
    long_threshold = int(config.get("long_target_char_threshold", 0))
    train_path = dataset_path(config, config["dataset"])
    eval_name = config.get("eval_dataset")
    eval_path = dataset_path(config, eval_name) if eval_name else None
    raw_train, encoded_train = load_encoded_dataset(
        tokenizer,
        train_path,
        cutoff_len,
        long_target_weight=long_weight,
        long_target_char_threshold=long_threshold,
    )
    raw_eval: list[dict[str, Any]] = []
    encoded_eval: list[dict[str, Any]] | None = None
    if eval_path:
        raw_eval, encoded_eval = load_encoded_dataset(
            tokenizer,
            eval_path,
            cutoff_len,
            long_target_weight=long_weight,
            long_target_char_threshold=long_threshold,
        )
    if args.max_train_samples:
        encoded_train = encoded_train[: args.max_train_samples]
    report = build_report(
        config_path,
        config,
        base_model,
        tokenizer,
        train_path,
        eval_path,
        encoded_train,
        encoded_eval,
    )

    load_kwargs = {
        "revision": revision,
        "trust_remote_code": False,
        "local_files_only": Path(base_model).exists(),
        "low_cpu_mem_usage": True,
    }
    dtype = torch.bfloat16 if config.get("bf16", True) else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            **load_kwargs,
        )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config.get("lora_rank", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config["lora_target"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class DataCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids: list[list[int]] = []
            attention_mask: list[list[int]] = []
            labels: list[list[int]] = []
            for feature in features:
                padding = max_len - len(feature["input_ids"])
                input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
                attention_mask.append(feature["attention_mask"] + [0] * padding)
                labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "sample_weights": torch.tensor(
                    [feature["sample_weight"] for feature in features],
                    dtype=torch.float32,
                ),
            }

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ):
            labels = inputs.pop("labels")
            sample_weights = inputs.pop("sample_weights")
            outputs = model(**inputs)
            shifted_logits = outputs.logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            token_losses = functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(shifted_labels.shape)
            mask = shifted_labels.ne(IGNORE_INDEX)
            per_sample = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            return (loss, outputs) if return_outputs else loss

    class MilestoneSaver(TrainerCallback):
        def __init__(self, milestones: dict[int, str]):
            self.milestones = milestones
            self.saved: set[str] = set()

        def on_step_end(self, callback_args, state, control, model=None, **kwargs):
            due = [
                (step, name)
                for step, name in sorted(self.milestones.items())
                if step <= state.global_step and name not in self.saved
            ]
            if due:
                step, name = due[-1]
                target = output_dir / "checkpoints" / name
                target.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(target, safe_serialization=True)
                (target / "checkpoint-info.json").write_text(
                    json.dumps(
                        {
                            "global_step": int(step),
                            "requested_epoch_fraction": next(
                                fraction
                                for fraction in fractions
                                if fraction_label(fraction) == name
                            ),
                            "note": "Adapter only; optimizer resume state is not included.",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self.saved.add(name)
                print(f"Saved adapter milestone {name} at optimizer step {step}")

    arguments_fields = TrainingArguments.__dataclass_fields__
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in arguments_fields
        else "evaluation_strategy"
    )
    micro_batch_size = int(config.get("per_device_train_batch_size", 1))
    accumulation_steps = int(config.get("gradient_accumulation_steps", 8))
    epochs = float(config.get("num_train_epochs", 2))
    estimated_steps_per_epoch = math.ceil(
        len(encoded_train) / (micro_batch_size * accumulation_steps)
    )
    estimated_total_steps = math.ceil(estimated_steps_per_epoch * epochs)
    warmup_steps = round(
        estimated_total_steps * float(config.get("warmup_ratio", 0.05))
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=encoded_eval is not None,
        **{eval_key: "steps" if encoded_eval is not None else "no"},
        eval_steps=int(config.get("eval_steps", 0)) or None,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=epochs,
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        optim=config.get("optim", "adamw_torch"),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
    )
    fractions = [float(value) for value in config.get("save_fractions", [0.5, 1, 1.5, 2])]

    def fraction_label(fraction: float) -> str:
        epoch = math.ceil(fraction)
        half = abs(fraction - epoch + 0.5) < 1e-8
        return f"epoch-{epoch}-half" if half else f"epoch-{epoch}"

    steps_per_epoch = math.ceil(
        len(encoded_train)
        / (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
    )
    total_steps = math.ceil(steps_per_epoch * float(training_args.num_train_epochs))
    milestones: dict[int, str] = {}
    for fraction in fractions:
        if fraction <= 0 or fraction > float(training_args.num_train_epochs):
            continue
        step = min(total_steps, max(1, round(steps_per_epoch * fraction)))
        milestones.setdefault(step, fraction_label(fraction))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_eval,
        data_collator=DataCollator(tokenizer.pad_token_id),
        callbacks=[MilestoneSaver(milestones)],
    )
    report["training"] = {
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": total_steps,
        "checkpoint_milestones": milestones,
        "config_parameters": {
            key: value
            for key, value in config.items()
            if key
            not in {
                "dataset_dir",
                "dataset",
                "eval_dataset",
                "model_name_or_path",
            }
        },
    }
    (output_dir / "training-audit-before.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    trainer.train()
    trainer.save_model(output_dir)
    report["training"]["completed"] = True
    report["training"]["final_global_step"] = int(trainer.state.global_step)
    (output_dir / "training-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved final adapter to {output_dir}")


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top-level YAML must be a mapping")
    if config.get("train_on_prompt") is not False:
        raise ValueError("train_on_prompt must be false: do not train on user prompts")
    if config.get("mask_history") not in {True, None}:
        raise ValueError("mask_history must be true")

    base_model = args.base_model or config["model_name_or_path"]
    tokenizer = load_tokenizer(base_model, config.get("model_revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cutoff_len = int(config.get("cutoff_len", 1024))
    long_weight = float(config.get("long_target_weight", 1.0))
    long_threshold = int(config.get("long_target_char_threshold", 0))
    train_path = dataset_path(config, config["dataset"])
    eval_name = config.get("eval_dataset")
    eval_path = dataset_path(config, eval_name) if eval_name else None
    _, encoded_train = load_encoded_dataset(
        tokenizer,
        train_path,
        cutoff_len,
        long_target_weight=long_weight,
        long_target_char_threshold=long_threshold,
    )
    encoded_eval = None
    if eval_path:
        _, encoded_eval = load_encoded_dataset(
            tokenizer,
            eval_path,
            cutoff_len,
            long_target_weight=long_weight,
            long_target_char_threshold=long_threshold,
        )

    if args.dry_run:
        report = build_report(
            config_path,
            config,
            base_model,
            tokenizer,
            train_path,
            eval_path,
            encoded_train,
            encoded_eval,
        )
        run_dry(report, encoded_train, encoded_eval)
        return

    train(config_path, config, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
