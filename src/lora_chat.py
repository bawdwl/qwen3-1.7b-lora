"""Minimal Qwen3 + PEFT adapter inference helper."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Mapping

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def adapter_directory(path: Path) -> Path:
    """Accept a PEFT directory or a training directory containing ``adapter``."""
    path = path.expanduser().resolve()
    if (path / "adapter_config.json").is_file():
        return path
    if (path / "adapter" / "adapter_config.json").is_file():
        return path / "adapter"
    raise FileNotFoundError(f"No adapter_config.json found under {path}")


class ChatEngine:
    """Load one Qwen model plus one or more selectable LoRA adapters."""

    def __init__(
        self,
        base_model: str | Path,
        adapters: Mapping[str, str | Path] | None = None,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        input_budget: int = 8192,
        seed: int = 42,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback is provided.")
        set_seed(seed)
        self.base_model = str(base_model)
        self.input_budget = input_budget
        self.lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            local_files_only=Path(self.base_model).exists(),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=torch_dtype,
            device_map=device_map,
            local_files_only=Path(self.base_model).exists(),
        ).eval()
        self.adapter_names: dict[str, str] = {}
        self.adapter_dirs: dict[str, Path] = {}
        if adapters:
            first = True
            for name, raw_path in adapters.items():
                path = adapter_directory(Path(raw_path))
                metadata = PeftConfig.from_pretrained(path)
                expected = Path(self.base_model).resolve()
                recorded = metadata.base_model_name_or_path
                if recorded and Path(recorded).exists() and Path(recorded).resolve() != expected:
                    raise ValueError(
                        f"Adapter {name} was trained for {recorded}, not {self.base_model}"
                    )
                peft_name = f"shiyu_{name}"
                if first:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        path,
                        adapter_name=peft_name,
                        is_trainable=False,
                    ).eval()
                    first = False
                else:
                    self.model.load_adapter(
                        str(path),
                        adapter_name=peft_name,
                        is_trainable=False,
                    )
                self.adapter_names[name] = peft_name
                self.adapter_dirs[name] = path
            self.model.eval()

    @property
    def available_modes(self) -> list[str]:
        return ["base", *self.adapter_names]

    def _select(self, mode: str):
        if mode == "base":
            return (
                self.model.disable_adapter()
                if isinstance(self.model, PeftModel)
                else nullcontext()
            )
        if mode not in self.adapter_names:
            raise ValueError(
                f"Unknown model mode {mode!r}; available: {', '.join(self.available_modes)}"
            )
        self.model.set_adapter(self.adapter_names[mode])
        return nullcontext()

    def _messages_with_budget(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Remove oldest complete turns when the chat exceeds the token budget."""
        system = [row for row in messages if row["role"] == "system"]
        dialogue = [row for row in messages if row["role"] != "system"]
        while True:
            candidate = system + dialogue
            text = self.tokenizer.apply_chat_template(
                candidate,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_count = len(
                self.tokenizer(text, add_special_tokens=False).input_ids
            )
            if token_count <= self.input_budget or len(dialogue) <= 1:
                return candidate
            dialogue = dialogue[2:] if len(dialogue) >= 2 else dialogue[-1:]

    def reply(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str = "base",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
    ) -> tuple[str, dict]:
        prepared = self._messages_with_budget(messages)
        text = self.tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        with self.lock:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with self._select(mode), torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
        new_tokens = generated[0, input_tokens:]
        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        return answer, {
            "model_mode": mode,
            "input_tokens": input_tokens,
            "output_tokens": int(new_tokens.numel()),
            "seconds": round(time.perf_counter() - started, 3),
        }
