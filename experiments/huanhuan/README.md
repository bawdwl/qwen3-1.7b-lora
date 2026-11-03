# Chat-Huanhuan experiment

该实验用于验证：

- Qwen3-1.7B 能否在 RTX 4060 8 GB 上完成 LoRA；
- 短单轮角色问答经过不同训练轮数后会怎样变化；
- 简短、核心和完整角色提示对复读与回答长度的影响。

## Data boundary

训练数据来自第三方 Chat-Huanhuan 项目，源头包含影视剧本对白。本仓库不复制原始数据、
衍生训练文件或 Adapter 权重。获取方式见 [`../../data/README.md`](../../data/README.md)。

## Fixed experiment

| 项目 | 值 |
|---|---:|
| Samples | 3,729 |
| Duplicate rows | 112, intentionally retained |
| Model | Qwen3-1.7B |
| LoRA | r8 / alpha32 / dropout0.1 |
| Target modules | q/k/v/o/gate/up/down |
| Cutoff | 384 |
| Micro batch | 1 |
| Gradient accumulation | 16 |
| Learning rate | `1e-4` |
| Epochs | 3 |
| Optimizer steps | 702 |
| Hardware | RTX 4060 Laptop, 8 GB |
| Training time | 1h53m36s |
| Final average training loss | about 2.98 |

No validation split was used because the upstream project does not provide one. This is a
reproduction limitation, not a claim that the selected checkpoint was statistically validated.

## Prompt comparison

| Prompt | Size | Best observed region | Main tradeoff |
|---|---:|---|---|
| Brief | 7 Token | half epoch | stable but very short |
| Core | 73 Token | epoch 1 | better balance and continuity |
| Character card | 298 Token | epoch 1 | stronger identity, shorter replies |

Across the 12 independent prompts and 8-turn conversation:

- half epoch under brief prompt had low repetition but repetitious “愿意陪皇上说会儿话” later;
- epoch 1 under core/card prompts was the best balance;
- epoch 2 and final increased exact or near repetition in several runs;
- a longer role card improved identity coverage but did not prevent brevity.

## Reproduction

1. Obtain `huanhuan.json` from the upstream project.
2. Run `scripts/prepare_huanhuan_author_close.py`.
3. Run `llamafactory-cli train configs/huanhuan_qwen3_1_7b_lora.yaml`.
4. Evaluate checkpoints:

```bash
python scripts/evaluate_checkpoints.py \
  --base-model models/qwen3-1.7b \
  --adapter-root outputs/huanhuan-qwen3-lora-v1 \
  --system-prompt prompts/character_card_zh.txt \
  --output-dir outputs/huanhuan-evaluation
```

The aggregate interpretation is included in
[`../../docs/FINETUNING_CASE_STUDY.md`](../../docs/FINETUNING_CASE_STUDY.md).
