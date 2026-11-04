# 裴既明：原创乙女向角色数据集

## 角色来源

裴既明是本项目为虚拟角色陪伴场景原创的虚构角色，不属于现实人物、影视角色或既有
二次元 IP。固定设定包括：

- 27 岁男性，现代半架空城市“临京”的裴氏资本执行董事；
- 用户是 28 岁女性、独立艺术顾问；
- 两人因私人藏品审核相识，关系处于有明确好感但尚未确定恋爱关系的阶段；
- 性格冷静、敏锐、克制，关心人的方式是记住细节和承担行动；
- 尊重拒绝、隐私和社交边界，不使用身份或金钱制造依赖。

完整设定见 [`character_bible.md`](character_bible.md)，训练和推理优先使用的精简提示见
[`system_prompt.txt`](system_prompt.txt)。

## 数据来源

这份数据集不是从网络抓取的真实聊天，也不是某个公开角色数据的二次清洗。制作流程是：

1. 由项目作者定义原创角色、关系和内容边界；
2. 编写 [`teacher_prompt.md`](teacher_prompt.md) 和场景要求；
3. 由 OpenCode `deepseek-v4.1-flash` 生成 120 个场景及 120 段 4–6 轮中文对话；
4. 额外生成 60 条长回复训练分支，用于缓解短回答退化；
5. 进行格式、重复、跨 split、长度、动作比例和内容边界审查；
6. 将每段多轮对话展开为一个“给定历史和系统提示，预测下一个角色回复”的 SFT 样本。

因此，数据反映的是“教师模型如何理解这个原创角色”，不代表真实用户聊天分布。
重新运行教师模型通常也不会得到逐字节相同的数据；可复现训练使用仓库中已经冻结的数据。

## 包含的原始数据

| 文件 | 内容 |
|---|---|
| `source/scenarios.json` | 120 个冻结场景，96 训练 / 13 开发 / 11 封存测试 |
| `source/dialogues.json` | 120 段完整多轮对白 |
| `source/long_reply_supplement.json` | 60 个长回复训练分支 |
| `source/dataset_audit.json` | 冻结时的自动审查结果 |

审查摘要：

- 556 条助手回复；
- 回复长度中位数 42 个汉字，P90 为 83，最大 120；
- 26.08% 的回复包含动作或神态；
- 2.34% 的回复以问句结束；
- 0 条完全重复助手回复；
- 0 条跨 train/development/test 近重复；
- 未发现禁用模板。

“审查通过”只表示满足这组程序规则，不表示角色质量或安全表现已经达到生产标准。

## assistant-only SFT

| 文件 | 来源会话 | assistant 目标 | 用途 |
|---|---:|---:|---|
| `sft/train.jsonl` | 156 个训练分支 | 577 | 参数更新 |
| `sft/validation.jsonl` | 13 段开发会话 | 54 | loss 和开发期检查点比较 |
| `sft/test.jsonl` | 11 段封存会话 | 45 | 选定检查点后只运行一次 |

每条 SFT 样本包含：

- `system_prompt`：训练时使用的精简角色提示；
- `messages`：当前可见的历史和本轮用户消息；
- 最后一条 `assistant`：唯一需要计算 loss 的角色回答；
- 会话、场景、类别和来源哈希等元数据。

训练脚本会屏蔽 system、user 和较早 assistant 历史的标签，只监督最后一条角色回答和
EOS。禁止随机按行拆分 train/validation/test，因为同一会话展开出的多个回复会泄漏。

## 训练配置

仓库中的
[`configs/pei_jiming_qwen3_1_7b_lora.yaml`](../../configs/pei_jiming_qwen3_1_7b_lora.yaml)
复现选中 Adapter 的第二轮实验：

- Qwen3-1.7B BF16；
- LoRA rank 4、alpha 8、dropout 0.05；
- 只训练 `q_proj / k_proj / v_proj / o_proj`；
- 2 epoch，micro batch 1，梯度累积 8，计划 146 个优化步；
- 学习率 `2e-5`，cosine，warmup 0.04；
- 最大长度 768；
- 60 字及以上的目标样本权重 3.0，并在 micro-batch 内归一化；
- 保存半轮、整轮、1.5 轮和 2 轮 Adapter。

完整命令、标签规则、监控和选模方法见
[`docs/REPRODUCTION.md`](../../docs/REPRODUCTION.md)。

## 结果摘要

开发集先选择检查点，封存测试随后只运行一次。开发集中 `epoch-1` 的审查结果为 0 个
硬失败，因此选入封存测试。封存测试中 LoRA 的 persona、context、naturalness 和
length-fit 平均分高于 Base，但仍比 Base 更短，并保留 1 个硬失败。

这只是小规模合成数据和单一 LLM 裁判的结果，不是人工盲评或生产质量证明。详细限制与
失败样本分析放在主要结果文档中。

## MNN Q8 发布

选中的第 1 轮 Adapter 已合并为 BF16 模型并以 MNN 3.5.0 Q8 block64 发布：

<https://modelscope.cn/models/Alixs666/shiyu-qwen3-1.7b-pei-jiming-mnn-q8>

本次发布已验证：

- Adapter 与合并模型的 next-token top-1 在 3/3 固定样本上一致；
- Q8 包可在 Windows CPU 的 MNN 3.5.0 中加载并正常结束生成；
- 尚未完成 Android 真机性能、温升和长对话复测。
