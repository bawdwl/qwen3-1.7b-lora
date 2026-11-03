# Qwen3-1.7b-lora

Qwen3-1.7B 单角色对话 LoRA 的可复现实验仓库。

这个项目关注的不是“训练 loss 能不能下降”，而是：

- LoRA 是否真的让模型更符合指定角色；
- 训练多少轮比较合适；
- 角色提示词应该写多长；
- 微调后为什么会出现回答变短、模式坍塌和复读；
- 1.7B 模型适合做到什么程度。

仓库提供数据、训练配置、推理入口、检查点评测脚本、失败复盘，以及两个原创角色
作者偏好的 PEFT LoRA Adapter。它不是完整的聊天 App，也不包含 Qwen3 基座权重。

## 已完成实验

| 实验 | 数据 | 训练 | 主要结论 |
|---|---|---|---|
| RoleBench 孙悟空 | 1,260 训练 / 146 验证 | 2 轮，316 步 | Base + 角色卡更自然；LoRA 过度简短 |
| 许澄 | 80 段训练 / 20 段验证；576 / 142 个回复目标 | 2 轮，144 步 | 作者偏好第 1 轮 + 核心提示 |
| 宋知意 | 64 / 8 / 8 段；276 / 35 / 35 个回复目标 | 2 轮，70 步 | 作者偏好第 1.5 轮 + 精简训练提示 |
| 甄嬛 | 3,729 条单角色问答 | 3 轮，702 步 | 第 1 轮附近最稳；后两轮出现短句复读 |
| 凉宫春日 | 1,009 训练 / 115 验证 | 2 轮，254 步 | 作者偏好第 1.5 轮 + 角色卡；质量门禁 No-Go |

五个实验共同表明：**LoRA 可以快速学习角色称谓和语气，但数据分布、提示词和选模方式
对最终质量的影响往往大于继续增加训练轮数。**

完整分析见
[`docs/FINETUNING_CASE_STUDY.md`](docs/FINETUNING_CASE_STUDY.md)。

## 仓库内容

```text
configs/       Qwen3-1.7B LoRA 训练配置
datasets/      许澄、宋知意原创角色数据集
prompts/       简短、核心和完整角色提示词
scripts/       数据准备、聊天、检查点评测和数据校验
src/           Qwen3 + PEFT 的小型推理库
docs/          实验记录、失败分析和提示词指南
weights/       两个原创角色作者偏好的 LoRA Adapter 和对应提示词
data/          Chat-Huanhuan、ChatHaruhi 第三方数据与许可说明
```

主要文档：

- [五组 LoRA 实验复盘](docs/FINETUNING_CASE_STUDY.md)
- [Qwen3-1.7B 微调与角色提示指南](docs/MODEL_AND_PROMPT_GUIDE.md)
- [甄嬛实验记录](docs/EXPERIMENT.md)
- [Chat-Huanhuan 复现边界](experiments/huanhuan/README.md)
- [ChatHaruhi 复现边界与结果](experiments/haruhi/README.md)
- [公开 Adapter 与本地 Base/LoRA 对照](weights/README.md)
- [同一提示词下的对照步骤](docs/LOCAL_COMPARISON.md)
- [对话截图命名与记录规范](screenshots/README.md)

## 数据集

本仓库包含两个原创虚构角色的教师模型合成数据：

| 角色 | 原始数据 | SFT 数据 |
|---|---|---|
| [许澄](datasets/xu-cheng/README.md) | 80 训练 / 20 验证 / 30 测试会话 | 576 训练 / 142 验证回复目标 |
| [宋知意](datasets/song-zhiyi/README.md) | 80 段完整对话 | 276 训练 / 35 验证 / 35 测试目标 |

数据包含角色设定、原始多轮对白、展开后的 assistant-only SFT 样本和审计结果。
所有对话均为虚构，不包含真实用户聊天记录。

文件行数和 SHA-256 记录在
[`datasets/manifest.json`](datasets/manifest.json)。

## 公开的 LoRA Adapter

仓库包含两个体积约 33 MiB 的 PEFT Adapter：

| 角色 | 检查点 | 配套提示词 | 说明 |
|---|---|---|---|
| 宋知意 | 第 1.5 轮 / 70 步中的第 52 步 | [`weights/song-zhiyi-epoch-2-half/system_prompt.txt`](weights/song-zhiyi-epoch-2-half/system_prompt.txt) | 作者体验偏好；求职 Demo 候选 |
| 许澄 | 第 1 轮 / 144 步中的第 72 步 | [`weights/xu-cheng-epoch-1/system_prompt.txt`](weights/xu-cheng-epoch-1/system_prompt.txt) | 作者体验偏好；仍是实验质量，不代表安全通过 |

Base 与 LoRA 必须使用同一提示词、同一历史、同一采样参数和同一基座权重。
具体命令见 [`docs/LOCAL_COMPARISON.md`](docs/LOCAL_COMPARISON.md)。

受底层素材或数据许可边界限制，甄嬛、凉宫春日和孙悟空 Adapter 不公开。
限制说明见 [`weights/README.md`](weights/README.md)。

## 快速开始

环境要求：

- Python 3.11
- CUDA 12.x
- 建议 8 GB 以上显存
- Windows 或 Linux

创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果不加载模型，只检查仓库中的数据：

```powershell
python scripts\validate_datasets.py
```

## 准备 Qwen3-1.7B

下载或复制 `Qwen/Qwen3-1.7B` 到：

```text
models/qwen3-1.7b/
```

也可以修改配置中的 `model_name_or_path`，直接使用 Hugging Face 或 ModelScope：

- [`configs/xu_cheng_qwen3_1_7b_lora.yaml`](configs/xu_cheng_qwen3_1_7b_lora.yaml)
- [`configs/song_zhiyi_qwen3_1_7b_lora.yaml`](configs/song_zhiyi_qwen3_1_7b_lora.yaml)
- [`configs/huanhuan_qwen3_1_7b_lora.yaml`](configs/huanhuan_qwen3_1_7b_lora.yaml)
- [`configs/haruhi_chatharuhi_qwen3_1_7b_lora.yaml`](configs/haruhi_chatharuhi_qwen3_1_7b_lora.yaml)

## 训练

许澄：

```bash
llamafactory-cli train configs/xu_cheng_qwen3_1_7b_lora.yaml
```

宋知意：

```bash
llamafactory-cli train configs/song_zhiyi_qwen3_1_7b_lora.yaml
```

甄嬛需要先自行获取第三方数据：

```bash
python scripts/prepare_huanhuan_author_close.py \
  --source data/source/huanhuan.json \
  --output-dir data/prepared/huanhuan-author-close-v1

llamafactory-cli train configs/huanhuan_qwen3_1_7b_lora.yaml
```

凉宫春日实验使用 ChatHaruhi 54K 聚合数据中固定版本的凉宫春日子集。数据不随仓库分发，
准备方式和许可边界见 [`experiments/haruhi/README.md`](experiments/haruhi/README.md)。

```bash
llamafactory-cli train configs/haruhi_chatharuhi_qwen3_1_7b_lora.yaml
```

统一训练配置：

- Qwen3 原生 `qwen3_nothink` 模板；
- 只监督 assistant 回答和 EOS；
- LoRA 七个注意力和 MLP 投影层；
- BF16、batch 1、梯度累积 8；
- 半轮和整轮检查点均保留。

## 对话

训练完成后，加载基础模型和指定 Adapter：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter outputs\lora-song-zhiyi-v2\checkpoints\epoch-2-half `
  --system-prompt datasets\song-zhiyi\sft\system_prompt.txt `
  --max-new-tokens 256
```

输入：

- `/reset` 清空当前对话；
- `/exit` 退出。

宋知意实验中最受作者偏好的组合是 **第 1.5 轮 + 精简训练提示**。

## 检查点评测

评测脚本要求 Adapter 目录中包含 `checkpoint-*`：

```text
outputs/lora-song-zhiyi-v2/
├── checkpoint-17/
├── checkpoint-35/
├── checkpoint-52/
└── checkpoint-70/
```

运行 Base 与各检查点的固定单轮、多轮对照：

```powershell
python scripts\evaluate_checkpoints.py `
  --base-model models\qwen3-1.7b `
  --adapter-root outputs\lora-song-zhiyi-v2 `
  --system-prompt datasets\song-zhiyi\sft\system_prompt.txt `
  --output-dir outputs\song-zhiyi-evaluation
```

输出：

- `results.jsonl`：全部原始回答、Token 和延迟；
- `summary.json`：角色称谓、过短回答、复读等统计；
- `report.md`：Base 与各检查点的对照。

自动统计不能代替人工盲评。

## 1.7B 模型适合什么

适合：

- 单角色语气和表达习惯适配；
- 低显存 LoRA 实验；
- 手机端量化 Demo；
- 固定角色、中等长度对话。

不适合单独承担：

- 高可靠安全判断；
- 长上下文和长期记忆；
- 复杂知识与推理；
- 强世界书约束下的多角色模拟。

每个角色单独训练 Adapter 的维护成本很高。面向多角色产品，更合适的是一个通用角色扮演
Adapter，再通过角色卡、世界书、检索记忆和安全规则区分角色。

## 角色提示词怎么写

经验建议：

| 层级 | 中文字符 | Qwen3 Token | 适用情况 |
|---|---:|---:|---|
| 极简身份 | 10–40 | 7–30 | LoRA 已稳定学会角色 |
| 核心人格 | 100–250 | 70–180 | 推荐默认 |
| 完整角色卡 | 350–800 | 250–600 | 复杂设定 |
| 超长档案 | 1,000+ Token | 不推荐 | 容易挤压回答和长对话历史 |

提示词至少应包含：

1. 身份和与用户的关系；
2. 性格以及不能出现的行为；
3. 称呼和说话方式；
4. 先回应用户当前内容；
5. 事实边界和不知道时的处理方式；
6. 禁止 AI 自述、机械反问和固定复读。

训练和部署应尽量使用同一个提示版本。

## 为什么微调后会复读

主要原因：

- 训练集回复普遍很短；
- 高频称谓、礼仪句和安慰模板被反复学习；
- 单角色数据量较小；
- 训练轮数过多；
- 推理时错误回答进入历史，后续继续被强化；
- 提示词规则不能替代训练数据中的真实 Token 分布。

因此不应默认训练 3 轮。更稳妥的流程是：

```text
Base + 最终提示词
→ 半轮 LoRA
→ 1 轮 LoRA
→ 1.5 轮 LoRA
→ 仅在有明确收益时继续训练
```

## 数据与许可

- 本仓库原创代码采用 MIT License；
- 许澄、宋知意数据是原创虚构角色的教师模型合成内容；
- 仓库包含许澄、宋知意两个原创角色的 LoRA Adapter，不包含 Qwen3 基座权重、
  合并模型、量化模型、API Key 或真实用户聊天记录；
- Chat-Huanhuan 数据源自影视剧本，第三方数据和权重未上传；
- ChatHaruhi 聚合数据卡与上游说明存在 CC-BY-4.0、CC-BY-NC-4.0 声明冲突，第三方数据、
  Adapter 和权重均未上传；
- MIT 只适用于本仓库原创代码，不代表第三方数据、基座模型或衍生模型获得相同授权；
- 使用第三方数据前请阅读 [`data/README.md`](data/README.md)。

## 验证

当前仓库包含：

- 数据行数、split 和 SFT 结构校验；
- Python 语法检查；
- Markdown 相对链接检查；
- GitHub Actions 基础校验。

运行：

```bash
python scripts/validate_datasets.py
python -m compileall -q src scripts
```
