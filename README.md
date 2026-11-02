# Qwen3-1.7b-lora

Qwen3-1.7B 单角色对话 LoRA 的可复现实验工程。

本项目包含：

- 将单角色问答数据转换为 LLaMA-Factory ShareGPT 格式；
- 在 RTX 4060 Laptop 8 GB 上运行的 BF16 LoRA 配置；
- 加载基础模型和任意 Adapter 检查点的终端聊天入口；
- 使用固定单轮问题与连续多轮对话比较 Base 和不同训练轮数的评测脚本；
- 数据集统计、训练记录和评测结论。

项目不包含模型权重、Adapter 权重、第三方原始语料或 API Key。

仓库包含两个本项目原创角色的教师模型合成对话数据集：

- [许澄数据集](datasets/xu-cheng/README.md)：100 段校准对话和 30 段隔离测试；
- [宋知意数据集](datasets/song-zhiyi/README.md)：80 段校园陪伴对话。

## 实验摘要

已验证的参考实验：

- 基座：Qwen3-1.7B
- LoRA：rank 8、alpha 32、dropout 0.1
- 目标层：`q/k/v/o/gate/up/down_proj`
- 训练数据：3,729 条，重复数据保留
- 最大长度：384 Token
- 微型 batch：1
- 梯度累积：16
- 学习率：`1e-4`
- epoch：3
- 优化步：702
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU 8 GB
- 训练时间：约 1 小时 54 分
- 最终平均训练 loss：约 2.98

关键结论：训练越久并不等于角色对话越好。当前检查点对比中，完整一轮且不使用
额外高频礼仪句的模型更平衡；后两轮出现了明显短句复读。详见
[`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)。

## 目录

```text
configs/                 LLaMA-Factory 训练配置
prompts/                 用于评测的简短提示、核心提示和角色档案
scripts/                 数据准备、聊天和检查点评测
src/                     小型独立推理库
datasets/                本项目原创角色的合成与 SFT 数据
data/README.md           第三方数据来源和使用边界
docs/                    复现步骤与实验结果
```

## 环境

推荐：

- Windows/Linux
- Python 3.11
- CUDA 12.x
- 8 GB 以上显存

```bash
python -m venv .venv
```

PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Bash：

```bash
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## 准备模型

下载或复制 `Qwen/Qwen3-1.7B` 到本地：

```text
models/qwen3-1.7b/
```

也可将
[`configs/huanhuan_qwen3_1_7b_lora.yaml`](configs/huanhuan_qwen3_1_7b_lora.yaml)
中的 `model_name_or_path` 改成 Hugging Face 或 ModelScope 模型 ID。

## 准备数据

第三方数据不随仓库分发。按 [`data/README.md`](data/README.md) 获取
Chat-Huanhuan 官方 `huanhuan.json`，并自行确认数据使用边界。

PowerShell：

```powershell
python scripts\prepare_huanhuan_author_close.py `
  --source data\source\huanhuan.json `
  --output-dir data\prepared\huanhuan-author-close-v1
```

Bash：

```bash
python scripts/prepare_huanhuan_author_close.py \
  --source data/source/huanhuan.json \
  --output-dir data/prepared/huanhuan-author-close-v1
```

脚本会保留全部重复行，不自动创建验证集或测试集。

## 训练

确认 YAML 中的数据和模型路径后运行：

```bash
llamafactory-cli train configs/huanhuan_qwen3_1_7b_lora.yaml
```

这个配置使用：

- `qwen3_nothink` 模板；
- 只监督 assistant 回答和 EOS；
- BF16 LoRA；
- 1×16 梯度累积；
- 每 117 个优化步保存检查点。

## 对话

默认加载最终 Adapter，并在没有系统提示时进行测试：

```bash
python scripts/chat_checkpoint.py \
  --base-model models/qwen3-1.7b \
  --adapter outputs/huanhuan-qwen3-lora-v1
```

加载指定检查点和角色档案：

```bash
python scripts/chat_checkpoint.py \
  --base-model models/qwen3-1.7b \
  --adapter outputs/huanhuan-qwen3-lora-v1/checkpoint-234 \
  --system-prompt prompts/character_card_zh.txt
```

## 评测

将需要比较的检查点目录放在同一输出根目录中：

```text
outputs/huanhuan-qwen3-lora-v1/
├── checkpoint-117/
├── checkpoint-234/
├── checkpoint-351/
├── checkpoint-468/
├── checkpoint-585/
└── checkpoint-702/
```

运行固定问题与 8 轮连续对话：

```bash
python scripts/evaluate_checkpoints.py \
  --base-model models/qwen3-1.7b \
  --adapter-root outputs/huanhuan-qwen3-lora-v1 \
  --system-prompt prompts/character_card_zh.txt \
  --output-dir outputs/evaluation
```

脚本会保存：

- `results.jsonl`：全部原始回答和 Token、延迟；
- `summary.json`：自动统计；
- `report.md`：检查点对比和典型回答。

自动指标只用于诊断，不能代替人工盲评。

## 实验复盘

不同数据集和训练轮数产生了不同结果：

- RoleBench 孙悟空训练后回答明显变短；
- 许澄在低损失之后仍出现安全边界、动作丢失和复读问题；
- 宋知意使用更贴近训练提示的精简版本后更稳定，作者最终偏好第 1.5 轮。

原因、证据和改进方向见
[`docs/FINETUNING_CASE_STUDY.md`](docs/FINETUNING_CASE_STUDY.md)。

## 数据与模型许可

本仓库原创代码使用 MIT License。

- Qwen3 基座和 Adapter 仍遵循上游模型许可；
- Chat-Huanhuan 数据源自影视剧本，项目说明包含学习研究与非商业限制；
- 许澄和宋知意数据是本项目原创角色的教师模型合成数据，不包含真实用户对话；
- 因此本仓库不附带数据、模型或 Adapter；
- 使用数据前请阅读 [`data/README.md`](data/README.md) 和
  [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)。
