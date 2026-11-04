# 许澄合成角色数据集

## 角色

许澄，女性，25 岁，常住厦门五年，在社区文化空间做夜班图书管理员。
她独居，有一只从救助站领养的橘猫“豆沙”，喜欢傍晚散步、城市声音、旧电影和简单做饭。
与用户关系起点是已经认识一段时间、可信但不黏人的朋友。

完整设定见 [`character_bible.md`](character_bible.md)。

## 数据来源

该角色是本项目原创虚构角色。全部对话由网页 GPT 系列教师模型根据 Character Bible、
场景卡和审核规则生成。原始记录中 60 个会话的 `teacher_model` 为 `GPT-6`，另有 70 个
早期会话只保存为 `unknown`；`teacher_version` 均不可确认。没有使用真实用户聊天记录。
数据经过场景、角色一致性、安全边界和格式审查后冻结。

## 原始多轮会话

| 文件 | 数量 | 用途 |
|---|---:|---|
| `source/train_sessions.jsonl` | 80 段 | 训练 |
| `source/validation_sessions.jsonl` | 20 段 | 验证 |
| `source/final_test_sessions.jsonl` | 30 段 | 隔离最终测试 |

训练和验证数据合计 100 段校准对话，覆盖日常、支持、连续性、自主性、安全与现实边界。

## assistant-only SFT

| 文件 | 数量 |
|---|---:|
| `sft/train.jsonl` | 576 个 assistant 目标 |
| `sft/validation.jsonl` | 142 个 assistant 目标 |

SFT 文件包含：

- 每条样本对应的完整 `system_prompt`；
- 当前样本可见的历史消息；
- 当前需要学习的 `assistant` 回复；
- 来源会话、场景、类别和验证信息。

训练配置：

- Qwen3-1.7B；
- LoRA rank 8、alpha 16、dropout 0.05；
- 七个注意力和 MLP 投影层；
- 2 epoch，72 个优化步/轮，共 144 步；
- BF16，最大长度 1024；
- 学习率 `5e-5`。

## 结果与限制

- 验证 loss：`6.0052 → 2.9014`；
- Base、半轮、一轮、1.5 轮和 2 轮权重均保留；
- 自动开发集曾因安全门禁选择第一轮半程；
- 作者后续直接对话时更偏好完整第 1 轮配合核心提示；
- 最终测试发现安全边界、凭证保存、欺骗建议、角色经历编造和过度简短等问题；
- 因此该数据集证明训练流程可复现，也暴露了教师模型合成数据与真实安全需求之间的差距。

更完整原因分析见仓库根目录
[`docs/FINETUNING_CASE_STUDY.md`](../../docs/FINETUNING_CASE_STUDY.md)。
