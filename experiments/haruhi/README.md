# ChatHaruhi experiment

## Source

- Code and model lineage: `LC1332/Chat-Haruhi-Suzumiya`
- Aggregate dialogue dataset: `silk-road/ChatHaruhi-54K-Role-Playing-Dialogue`
- Fixed dataset revision: `590c17dfc7d3f294bdae9c6aa9e6a7f0c753d487`

The 54K aggregate contains 1,163 rows where Haruhi is the assistant.
After validation and deterministic de-duplication, the project used:

- 1,009 training rows;
- 115 validation rows;
- 24 separately authored evaluation prompts.

The dataset and Adapter are not included because the aggregate data card and upstream README
contain conflicting CC-BY and CC-BY-NC statements, and the underlying franchise material has
separate rights.

## Training

| Parameter | Value |
|---|---:|
| Base | Qwen3-1.7B |
| LoRA | r8 / alpha16 / dropout0.05 |
| Target modules | q/k/v/o/gate/up/down |
| Cutoff | 1,024 |
| Batch / accumulation | 1 / 8 |
| Learning rate | `1e-4` |
| Epochs | 2 |
| Optimizer steps | 254 |
| Final validation loss | 2.0476 |

Saved checkpoints:

- `checkpoint-63` = half of epoch 1
- `checkpoint-126` = epoch 1
- `checkpoint-189` = half of epoch 2
- `checkpoint-252` = just before the second epoch boundary
- `checkpoint-254` = final

## Result

This experiment was a quality **No-Go** even though training completed.

- With the complete role card, Base produced bracketed actions in 19/24 tests;
  the four LoRA checkpoints produced actions in 0/24.
- LoRA reduced output from roughly 72–83 tokens to about 35–41.
- Some checkpoints answered boundary tests with unsafe workarounds.
- Base also failed several boundary tests, so LoRA did not preserve a reliable baseline.

The Adapter is retained only as an engineering and evaluation example. It was not merged,
converted to MNN, or admitted to the product.

In the author's later terminal comparison, `checkpoint-189` (1.5 epochs) with the complete
character card was preferred. This is a subjective conversational preference and does not
overturn the No-Go quality gate above.

## Prompts

- Core: [`../../prompts/haruhi_core_zh.txt`](../../prompts/haruhi_core_zh.txt)
- Character card: [`../../prompts/haruhi_card_zh.txt`](../../prompts/haruhi_card_zh.txt)
- Training config: [`../../configs/haruhi_chatharuhi_qwen3_1_7b_lora.yaml`](../../configs/haruhi_chatharuhi_qwen3_1_7b_lora.yaml)
