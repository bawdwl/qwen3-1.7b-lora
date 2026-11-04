"""Check the local Qwen3 LoRA training environment without loading model weights."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def model_report(model_path: Path | None) -> dict:
    if model_path is None:
        return {"provided": False}
    path = model_path.expanduser().resolve()
    required = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    weight_files: list[str] = []
    single_weight = path / "model.safetensors"
    index_file = path / "model.safetensors.index.json"
    if single_weight.is_file():
        weight_files = [single_weight.name]
    elif index_file.is_file():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
            weight_files = sorted(set(index.get("weight_map", {}).values()))
            missing.extend(
                name for name in weight_files if not (path / name).is_file()
            )
        except Exception as exc:
            missing.append(f"model.safetensors.index.json ({exc})")
    else:
        missing.append("model.safetensors or model.safetensors.index.json")
    report: dict = {
        "provided": True,
        "path": str(path),
        "exists": path.is_dir(),
        "missing_required_files": missing,
        "weight_files": weight_files,
    }
    if missing or not path.is_dir():
        return report
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=False,
            local_files_only=True,
        )
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with OK."},
        ]
        enabled = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        disabled = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        report.update(
            {
                "tokenizer_class": tokenizer.__class__.__name__,
                "chat_template": True,
                "thinking_switch_supported": True,
                "thinking_template_tokens": len(
                    tokenizer(enabled, add_special_tokens=False).input_ids
                ),
                "nonthinking_template_tokens": len(
                    tokenizer(disabled, add_special_tokens=False).input_ids
                ),
                "eos_token": tokenizer.eos_token,
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": sys.version_info[:2] == (3, 11),
        },
        "platform": platform.platform(),
        "packages": {
            name: package_version(module)
            for name, module in {
                "torch": "torch",
                "transformers": "transformers",
                "peft": "peft",
                "accelerate": "accelerate",
                "safetensors": "safetensors",
                "yaml": "yaml",
            }.items()
        },
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1),
        "model": model_report(args.model),
    }

    try:
        import torch

        cuda = torch.cuda.is_available()
        report["cuda"] = {
            "available": cuda,
            "device_count": torch.cuda.device_count() if cuda else 0,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda else False,
        }
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            report["cuda"].update(
                {
                    "device_name": properties.name,
                    "vram_total_gb": round(properties.total_memory / 1024**3, 1),
                }
            )
    except Exception as exc:
        report["cuda"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    failures: list[str] = []
    if not report["python"]["supported"]:
        failures.append("Python 3.11 is required by this tested recipe")
    if not report["cuda"]["available"]:
        failures.append("CUDA is unavailable")
    if report["cuda"].get("available") and not report["cuda"].get("bf16_supported"):
        failures.append("This GPU/PyTorch build does not report BF16 support")
    for package in ("torch", "transformers", "peft", "accelerate", "yaml"):
        if not report["packages"][package]:
            failures.append(f"Missing package: {package}")
    if report["model"].get("provided"):
        if not report["model"].get("exists"):
            failures.append("Model directory does not exist")
        if report["model"].get("missing_required_files"):
            failures.append("Model directory is missing required files")
        if report["model"].get("error"):
            failures.append("Model tokenizer/chat template could not be loaded")
    report["failures"] = failures
    report["ok"] = not failures

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {report['python']['version']} ({report['python']['executable']})")
        print(f"Torch: {report['packages']['torch']}")
        print(f"Transformers: {report['packages']['transformers']}")
        print(f"PEFT: {report['packages']['peft']}")
        print(f"CUDA: {report['cuda'].get('available')}")
        if report["cuda"].get("device_name"):
            print(
                f"GPU: {report['cuda']['device_name']} "
                f"({report['cuda']['vram_total_gb']} GiB)"
            )
            print(f"BF16: {report['cuda'].get('bf16_supported')}")
        if report["model"].get("provided"):
            print(
                f"Model: {report['model'].get('path')} "
                f"(ready={report['model'].get('exists') and not report['model'].get('error') and not report['model'].get('missing_required_files')})"
            )
        if failures:
            print("\nFailures:")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("\nEnvironment check passed.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
