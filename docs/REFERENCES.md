# 参考的官方文档与高星项目

本项目没有把某一个角色项目的训练配方当作通用答案。通用流程主要参考以下官方文档、
论文和成熟开源项目，再结合 RTX 4060 Laptop 8 GB 的真实训练结果收敛成本仓库脚本。

## Qwen3 官方资料

- [Qwen3-1.7B 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [Qwen 文档](https://qwen.readthedocs.io/)

本项目采用的原则：

- Qwen3 需要 `transformers>=4.51.0` 才能正确识别 `qwen3` 架构；
- Qwen3 支持 thinking/non-thinking 切换；
- 对话训练和推理必须使用相同的 Chat Template 和 thinking 设置；
- 角色聊天和低延迟应用通常使用 `enable_thinking=False`；
- 采样参数不能原样照搬其他模型，Qwen3 模型卡对 thinking/non-thinking 给出了不同建议。

## Hugging Face PEFT

- [PEFT GitHub](https://github.com/huggingface/peft)
- [PEFT 文档](https://huggingface.co/docs/peft/)
- [LoRA 原始论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)

本项目采用的原则：

- LoRA 冻结基座，只训练低秩 Adapter；
- Adapter 可以很小，但训练时仍需加载完整 BF16 基座和激活；
- rank、alpha、target modules 是实验变量，不应只追求更大 rank；
- 低显存场景可以进一步使用 QLoRA，但本仓库内置脚本只实现 BF16 LoRA。

## Hugging Face TRL

- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [TRL 数据集格式](https://huggingface.co/docs/trl/main/en/dataset_formats)

本项目采用的原则：

- SFT 是最常用的指令/对话适配方法；
- 对话数据应保留明确的 system/user/assistant 角色；
- assistant-only loss 需要额外的 mask，不能默认所有 Token 都训练；
- packing 能提高吞吐，但会改变序列边界和截断行为，小规模对照实验默认关闭；
- 必须同时报告 loss、重复、长度、角色/任务质量，而不是只看验证 loss。

## LLaMA-Factory

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA-Factory 中文文档](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)
- [ShareGPT 数据集说明](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)

本项目采用的原则：

- 自定义数据集必须在 `dataset_info.json` 中注册；
- ShareGPT 消息顺序必须严格为 user/assistant；
- 训练模板、推理模板和 thinking 设置必须一致；
- LLaMA-Factory 默认可能训练历史 assistant 回答；需要 assistant-only 时必须在框架配置
  或数据预处理中明确屏蔽；
- 该框架适合需要 QLoRA、多 GPU、可视化和大规模调参的用户；本仓库提供不依赖该框架的
  最小可运行实现，便于检查每一步。

## 本项目没有直接复制的内容

- 没有复制 LLaMA-Factory 的代码或配置；
- 没有把某个角色 LoRA 的超参数当成通用最优值；
- 没有承诺任何 rank、学习率或 epoch 在所有数据上有效；
- 没有把验证 loss 当成最终质量分数。
