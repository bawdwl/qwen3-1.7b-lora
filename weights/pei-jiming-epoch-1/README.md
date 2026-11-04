# 裴既明 LoRA

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- LoRA: rank 4, alpha 8, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Selected from: second training recipe, epoch 1
- Adapter SHA-256:
  `82A3AE025EBB27ED224A1042375B59355E8A945CDF6D2A5BC5F4EDF777A560CE`

Use `system_prompt.txt` for the reproduction comparison. The Adapter is a small
single-character experiment, not a merged model or production safety claim.

`selection.json` and `run-config.json` preserve the checkpoint-selection result
and tested training environment.
