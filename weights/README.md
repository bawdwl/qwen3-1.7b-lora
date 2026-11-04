# Released LoRA adapters

This directory contains PEFT LoRA adapters trained from original fictional
characters whose dialogue data is included in this repository.

| Character | Adapter | Training point | Prompt | Status |
|---|---|---|---|---|
| [宋知意](../datasets/song-zhiyi/README.md) | `song-zhiyi-epoch-2-half/` | 1.5 epochs / step 52 of 70 | `system_prompt.txt` | Author-preferred conversational result; Demo candidate |
| [许澄](../datasets/xu-cheng/README.md) | `xu-cheng-epoch-1/` | 1 epoch / step 72 of 144 | `system_prompt.txt` | Author-preferred conversational result; experimental quality, not a safety pass |
| [裴既明](../datasets/pei-jiming/README.md) | `pei-jiming-epoch-1/` | Epoch 1 of a 146-step run | `system_prompt.txt` | Development-selected result; known shorter-response regression |

The 宋知意 and 许澄 adapters use:

- base model: `Qwen/Qwen3-1.7B`;
- PEFT LoRA rank 8, alpha 16, dropout 0.05;
- seven projection modules: `q/k/v/o/gate/up/down`;
- about 33 MiB per adapter, excluding the base model and optimizer states.

The 裴既明 adapter uses rank 4, alpha 8 and attention projections only because
the earlier higher-capacity recipe collapsed toward short answers. It is about
6.2 MiB.

The base model weights are not included. Load the pinned base model yourself
and supply it with `--base-model`.

## Not released

The repository intentionally excludes:

- ChatHaruhi / 凉宫春日 adapters because the upstream license statements
  conflict and the franchise has separate rights;
- RoleBench 孙悟空 adapters because the generated-answer dataset's derivative
  redistribution boundary is not treated as automatically clear;
- optimizer states, tokenizer copies and merged full-model weights.

## Minimal reproduction

Use the same prompt, history, generation settings and base model for the Base
and LoRA runs. Reset the conversation before each fixed comparison.

```powershell
# Base with the same prompt
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256

# Song Zhiyi LoRA
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter weights\song-zhiyi-epoch-2-half `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256
```

For Xu Cheng or Pei Jiming, replace the adapter directory with
`weights\xu-cheng-epoch-1` or `weights\pei-jiming-epoch-1`.

See [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for dataset preparation,
assistant-only labels, full training commands and checkpoint selection.

## License

The Adapter files are released subject to the Apache-2.0 license of
`Qwen/Qwen3-1.7B` and the provenance of the original synthetic character data
documented in this repository. The repository's MIT License continues to apply
to original source code, not as a replacement for upstream model licenses.
