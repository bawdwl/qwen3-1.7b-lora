# Qwen3-1.7B LoRA 通用复现指南

这不是某个角色专用的训练说明书。本文从零说明如何为任意对话、问答或任务数据训练
Qwen3-1.7B LoRA，包括环境安装、数据选择与清洗、参数选择、训练、监控、检查点比较、
推理和常见故障。仓库中的三个角色只是可直接运行的案例。

先给出最重要的结论：

- SFT 的输入必须是“提示与目标回答”，而不是随便下载的纯文本；
- 每一轮必须只监督目标 assistant，不训练 system 和用户问题；
- train/validation/test 要按完整会话切分；
- rank、学习率和 epoch 是实验变量，不能只看 loss 选；
- Qwen3 的训练和推理必须使用一致 Chat Template 及 thinking 设置；
- 8 GB 显存可以从 rank 4/8、batch 1、梯度累积 8、BF16 LoRA 开始。

## 1. 训练目标与 LoRA 适用范围

全参数微调会更新 Qwen3-1.7B 的约 17 亿参数和 Adam 优化器状态。LoRA 冻结基座，只在部分
线性层旁边训练小矩阵。PEFT 官方文档说明，这种方式显著降低显存、存储和每个任务需要
保存的权重体积；本项目的 Adapter 只有数 MiB 到数十 MiB。

适合 LoRA：

- 改变回答风格、格式、角色口吻或业务流程；
- 单领域问答和指令跟随；
- 让小模型更稳定地遵守已有能力范围；
- 在消费级 GPU 上做小规模实验。

不适合只靠 LoRA 解决：

- 让 1.7B 从零掌握大量新事实；
- 替代 RAG、数据库或工具调用；
- 用几百条数据获得可靠医学、法律或金融判断；
- 以训练 loss 代替产品指标和安全评测；
- 多角色产品为每个角色无限增加独立 Adapter。

## 2. 选择正确基座和聊天模式

### 2.1 使用 `Qwen/Qwen3-1.7B`

本仓库使用已完成后训练的通用 Chat 模型：

```text
Qwen/Qwen3-1.7B
revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

不要用以下模型冒充同一实验：

- `Qwen3-1.7B-Base`：缺少同等的指令对话后训练；
- GPTQ/AWQ/MNN 量化模型：数值格式不同；
- `Qwen3-0.6B` 或 `4B`：容量和 Tokenizer/结构不同；
- 其他 revision：Chat Template 或模型行为可能变化。

模型卡要求 Transformers 至少支持 `qwen3` 架构。官方当前建议使用兼容的新版本；本项目在
`requirements.txt` 固定了历史训练版本。

### 2.2 thinking 与 non-thinking

Qwen3 支持两种模式：

- `enable_thinking=True`：先产生思考内容，适合复杂推理；
- `enable_thinking=False`：直接回答，适合角色聊天、分类、抽取、客服和低延迟任务。

训练与推理要保持一致。角色数据通常不含 `<think>` 轨迹，本项目配置使用
`qwen3_nothink`，`train_lora.py` 通过 `enable_thinking=False` 应用 Chat Template。

如果误用 thinking 模式训练不含思考过程的数据，模型可能学到空思考块、额外标记或不稳定
输出。若目标是推理任务，应准备包含完整思考轨迹的数据，并在推理端保留思考解析。

## 3. 安装并验证环境

### 3.1 推荐硬件

本项目验证过：

- NVIDIA RTX 4060 Laptop GPU，8 GB 显存；
- 单卡 BF16 LoRA；
- 输入最大约 1,024 Token；
- Qwen3-1.7B 完整 BF16 基座；
- 约 4.3–4.6 GiB 峰值已分配显存，操作系统的 CUDA context 还会占额外显存。

推荐最低配置：

| 项目 | 最低 | 更稳妥 |
|---|---:|---:|
| GPU 显存 | 8 GB | 12–16 GB |
| 系统内存 | 16 GB | 32 GB |
| 空闲磁盘 | 10 GB | 20 GB |
| Python | 3.11 | 3.11 |
| CUDA | 12.x | 与 PyTorch wheel 完全匹配 |

低于 6 GB 时，本仓库的 BF16 LoRA 路径很可能无法运行。可选择 LLaMA-Factory 或 TRL 的
QLoRA 路径，但这会产生另一种实验条件，不应与 BF16 结果直接排名。

先确认 NVIDIA 驱动可见：

```bash
nvidia-smi
```

必须看到 GPU 名称、驱动和 CUDA 信息。若命令不存在，先安装与 GPU 匹配的 NVIDIA 驱动，
并确认 `python -m pip show torch` 指向包含 CUDA 的 wheel。

### 3.2 Windows

```powershell
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

安装与机器匹配的 CUDA PyTorch：

```powershell
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

`cu126` 是历史环境。机器驱动或 CUDA 不同时，应根据 PyTorch 官方 selector 替换 wheel，
不要把 CPU wheel 混进训练环境。

如果 PowerShell 阻止虚拟环境激活，可在当前进程临时解除脚本限制：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 3.3 Linux

```bash
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### 3.4 一行检查

```powershell
python scripts\check_environment.py `
  --model models\qwen3-1.7b
```

检查内容：

- Python 版本；
- PyTorch/Transformers/PEFT/Accelerate/Safetensors/PyYAML；
- CUDA 是否可用；
- GPU 名称、总显存和 BF16；
- 当前磁盘剩余空间；
- Qwen3 模型目录必需文件；
- Tokenizer 和 thinking/non-thinking Chat Template 是否可应用。

模型下载前可以先不传 `--model`，只检查 GPU 和包；下载后必须再带模型路径运行一次。

## 4. 下载 Qwen3-1.7B

模型固定放在：

```text
models/qwen3-1.7b/
```

该路径在 `.gitignore` 中，权重不会进入 Git。

Hugging Face：

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen3-1.7B ^
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ^
  --local-dir models/qwen3-1.7b
```

Linux 把 `^` 换成 `\`。如果安装的是旧 CLI：

```bash
huggingface-cli download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/qwen3-1.7b
```

ModelScope：

```bash
python -m pip install modelscope
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/qwen3-1.7b
```

最低文件：

```text
config.json
generation_config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
```

目录约 4.1 GB，不是 410 MB。下载中断后必须检查 `.safetensors` 文件和 index 是否完整。

## 5. 选择与准备训练数据

### 5.1 先判断任务需要什么数据

| 目标 | 应选数据 | 不应使用 |
|---|---|---|
| 角色聊天 | 多轮 user/assistant 对话 | 小说原文直接当聊天 |
| 客服/业务问答 | 真实授权工单、FAQ、状态与操作边界 | 只含产品宣传文案 |
| 分类/抽取 | 输入和标准输出对 | 只有无标签文章 |
| 代码/工具调用 | 明确工具 schema 和成功轨迹 | 只凭自然语言描述工具 |
| 继续预训练 | 大规模 raw text | 几百条聊天问答 |

LoRA SFT 不是继续预训练。纯文本可以使用 `text` 字段做因果语言建模，但本仓库内置训练器
面向对话 SFT，要求 system/user/assistant 结构。

### 5.2 数据集来源优先级

1. 自己拥有再分发和使用权的产品数据；
2. 有明确许可证、贴近目标任务的公开数据集；
3. 按规范生成并完成抽检与过滤的教师模型数据；
4. 没有来源、自动洗稿、模板同质化或含真实隐私的抓取数据。

使用公开数据集前检查：

- dataset card 与仓库许可证；
- 原始内容是否来自第三方 IP；
- 数据是否含个人、医疗、金融或账号信息；
- train/test 是否已划分，能否判断污染；
- 对话角色是否一致；
- 是否允许训练和再分发 Adapter。

Hugging Face 和 ModelScope 的下载地址不能代替许可证审查。仓库提供的三个角色数据均为
原创教师模型合成的虚构对话，只是可运行示例，不代表真实用户分布。

可选数据集类型示例（只作为检索起点，使用前必须核对当前 dataset card、条款和下游限制）：

| 类型 | 常见候选 | 适合先验证什么 | 主要风险 |
|---|---|---|---|
| 通用英文多轮对话 | `HuggingFaceH4/ultrachat_200k` | 对话格式、assistant-only SFT 流程 | 合成分布、语言不完全匹配 |
| 中文日常多轮对话 | LCCC 系列 | 中文长短回复和日常聊天分布 | 数据较旧、版本与许可需核对 |
| 中文自然对话 | NaturalConv | 中文自然轮次和话题变化 | 获取渠道及再分发条款需核对 |
| 知识主题对话 | KdConv | 主题知识与多轮承接 | 偏电影/音乐/旅游，非通用陪伴 |
| 大规模通用指令 | Infinity-Instruct 等 | 通用指令覆盖和大数据训练流程 | 体量大、来源混合，需严格过滤 |

不要把所有候选直接拼成一个训练集。先分别抽样，统一角色名、system prompt、时间格式和
输出长度，再计算跨集近似重复。真实产品数据与教师模型数据应记录不同来源标签，方便分析
模型到底学到了哪一部分。

### 5.3 推荐目录

```text
datasets/my-task/
├── sft/
│   ├── dataset_info.json
│   ├── system_prompt.txt
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── source/
    └── sessions.jsonl
```

`source/` 保存完整多轮会话，便于按会话切分；`sft/` 保存展开后的训练目标。

### 5.4 SFT 样本

每行是一个 JSON：

```json
{
  "id": "session-001:a01",
  "session_id": "session-001",
  "system_prompt": "你是[customer service]，先直接回答问题...",
  "messages": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "目标回答"}
  ]
}
```

多轮样本可以包含多个 user/assistant 对，但最后一条必须是当前要预测的 assistant。

### 5.5 展开完整会话

一段 4 轮对话应展开为 4 条样本：

1. user1 → assistant1；
2. user1/assistant1/user2 → assistant2；
3. 前两轮历史 + user3 → assistant3；
4. 前三轮历史 + user4 → assistant4。

仓库提供通用展开器：

```powershell
python scripts\build_sft.py `
  --input path\to\sessions.jsonl `
  --output path\to\sft\train.jsonl `
  --system-prompt path\to\system_prompt.txt
```

JSON 数组输入可指定：

```powershell
python scripts\build_sft.py `
  --input path\to\dialogues.json `
  --input-key dialogues `
  --output path\to\sft\train.jsonl `
  --system-prompt path\to\system_prompt.txt
```

脚本会验证角色交替和目标结构，但不会替你完成内容审查或 split。

### 5.6 切分规则

必须在展开 SFT 前按完整会话切分：

- train：训练参数；
- validation/dev：训练期间选学习率、epoch 和检查点；
- test：所有选择结束后只运行一次。

禁止随机按单行切分，因为同一条会话展开出的样本含有大量重叠历史。禁止用 base 模型生成
测试答案后再把相同输入放进训练集。

小数据集可使用大致 80/10/10 或 64/8/8 的会话划分，但要优先保证场景覆盖。测试集至少要有
角色/任务、多轮承接、不知道、不应执行、自然结束等困难样本。

### 5.7 数据清洗

必须检查：

- user/assistant 严格交替，最后一条是 assistant；
- 空回答、重复回答、乱码、控制 Token；
- system prompt 版本是否混杂；
- 同一问题复制几十次；
- 输出长度是否全是一种长度；
- 用户问题是否被误写进 assistant 目标；
- 角色名、身份、日期和事实是否冲突；
- train/validation/test 是否存在近重复；
- 是否含手机号、地址、账号、API Key、病历等敏感信息。

数据量不是唯一标准。200 条覆盖广、无泄漏的数据通常比 2,000 条模板回复更值得训练。

## 6. assistant-only 标签

对下面的对话：

```text
System: 你是 X
User: 今天很累
Assistant: 那就先休息一下
```

如果整段都计算 loss，模型会学习：

- 如何生成“你是 X”；
- 如何生成用户问题；
- 如何从上下文中复制文本。

正确目标是让 labels 类似：

```text
System tokens       -100 -100 ...
User tokens         -100 -100 ...
Target tokens       “那就先休息一下” + EOS
```

本仓库 `train_lora.py` 的步骤：

1. 渲染 system 和最终 assistant 之前的上下文，并加入 assistant generation prompt；
2. 将目标回答和 EOS 接在后面；
3. 分别 Tokenize 前缀和完整文本，检查前缀 Token 完全一致；
4. 前缀 labels 全部设为 `-100`；
5. 只保留目标回答和 EOS 的真实 Token ID；
6. 超长时移除最早的完整 user/assistant 对；
7. 如果 system、最终问题和目标本身仍超长，直接报错。

先用 dry-run 检查：

```powershell
python scripts\train_lora.py `
  --config configs\qwen3_1_7b_lora_template.yaml `
  --base-model models\qwen3-1.7b `
  --dry-run
```

应在训练前改好模板中的数据名和路径，或直接对已有配置运行。

## 7. 参数怎么选

### 7.1 LoRA 参数

下表是 8 GB 显存、1.7B 模型、小到中等数据集的起点，不是普遍最优值。

| 参数 | 建议起点 | 调整方法 |
|---|---:|---|
| `lora_rank` | 8 | rank 4 更轻；rank 16 只在明显欠拟合时比较 |
| `lora_alpha` | 16 | 常从 1×–2× rank 开始；改动后重做对照 |
| `lora_dropout` | 0.05 | 训练 loss 很低但验证变差时可试 0.1 |
| target | q/k/v/o | 显存紧时 q/v；复杂领域再考虑加 MLP |
| 可训练参数 | 0.03%–1% | 不是越少越省越好，要观察是否欠拟合 |

LoRA 的缩放包含 alpha/rank。只提高 rank 而固定 alpha，会改变有效更新强度，因此 rank
和 alpha 应成组记录和比较。

### 7.2 学习率

先用保守值：

| 训练目标数 | 起始学习率 |
|---|---:|
| < 500 | `1e-5` 或 `2e-5` |
| 500–2,000 | `2e-5` 或 `5e-5` |
| 2,000–10,000 | `5e-5` |
| > 10,000 | `5e-5`–`1e-4` |

学习率过高常见表现：

- 回答快速变短；
- 只会固定口头禅；
- 复读用户问题；
- 普通能力退化；
- 输出不结束。

学习率过低常见表现：

- 1–2 轮后风格变化很小；
- train/validation loss 下降但仍像 Base；
- 角色知识、格式或任务行为没有学会。

### 7.3 epoch 和优化步

```text
steps_per_epoch = ceil(训练样本数 / (batch × 梯度累积 × GPU 数))
total_steps = steps_per_epoch × epoch
```

三个仓库例子的计划值：

| 配置 | 训练目标 | batch × 累积 | 每轮步数 | 2 轮总步数 |
|---|---:|---:|---:|---:|
| 宋知意 | 276 | 1 × 8 | 35 | 70 |
| 许澄 | 576 | 1 × 8 | 72 | 144 |
| 裴既明 | 577 | 1 × 8 | 73 | 146 |

建议策略：

| 数据规模 | 第一轮比较 |
|---|---|
| < 500 目标 | 0.5、1、1.5 轮 |
| 500–2,000 目标 | 1、1.5、2 轮 |
| 2,000–10,000 目标 | 0.5、1、1.5 轮 |
| > 10,000 目标 | 先 0.5、1 轮 |

这里的“数据规模”是展开后的 assistant 目标数，不是原始对话段数。

### 7.4 batch 和梯度累积

8 GB 显存推荐：

```text
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

有效 batch = micro batch × 梯度累积 × GPU 数。不要为了追求有效 batch 把 micro batch
调大后 OOM。数据量少时可以保留 batch 1 和梯度累积 8，训练更慢但更稳。

### 7.5 上下文长度

`cutoff_len` 应按 Token 而不是字符设置：

1. dry-run 查看输入 Token 的 P95/P99/max；
2. 选 P95 或 P99；
3. 留约 15%–25% 余量；
4. 若少数超长样本包含关键信息，可单独增加长度或删减旧历史。

本仓库 `cutoff_len` 是总预算，包含 system、历史、最终 user、目标 assistant 和 EOS。
不要静默截掉监督目标。

### 7.6 其他参数

```text
learning_rate: 2e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
seed: 42
```

- cosine：小到中等数据常用；
- 5% warmup：降低最初更新不稳定风险；
- max grad norm 1.0：抑制极端梯度；
- BF16：A100/40 系等支持时优先；
- gradient checkpointing：牺牲少量速度换取显存；
- 固定 seed 只能提高可重复性，不能保证逐 bit 一致。

## 8. 开始训练并监控

### 8.1 使用通用模板

```powershell
Copy-Item `
  configs\qwen3_1_7b_lora_template.yaml `
  configs\my_task_qwen3_1_7b_lora.yaml
```

编辑以下字段：

```yaml
model_name_or_path: models/qwen3-1.7b
dataset_dir: datasets/my-task/sft
dataset: my_task_train
eval_dataset: my_task_validation
output_dir: outputs/lora-qwen3-1.7b-my-task
```

`dataset_dir/dataset_info.json` 示例：

```json
{
  "my_task_train": {
    "file_name": "train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "system": "system_prompt"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "my_task_validation": {
    "file_name": "validation.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "system": "system_prompt"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 8.2 先 dry-run

```powershell
python scripts\train_lora.py `
  --config configs\my_task_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b `
  --dry-run
```

确认：

- 行数正确；
- 输入 Token 最大不超过 cutoff；
- 每行 supervised tokens > 0；
- train/validation 没有同一 session；
- system prompt 只有一个预期版本。

### 8.3 小规模 GPU smoke

```powershell
python scripts\train_lora.py `
  --config configs\my_task_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b `
  --output-dir outputs\smoke-my-task `
  --max-train-samples 20
```

smoke 只验证模型、CUDA、loss、保存和验证逻辑，不能评价质量。它会读取完整验证集。

### 8.4 正式训练

```powershell
python scripts\train_lora.py `
  --config configs\my_task_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b
```

输出：

```text
outputs/lora-qwen3-1.7b-my-task/
├── adapter_config.json
├── adapter_model.safetensors
├── training-audit-before.json
├── training-audit.json
├── training_args.bin
└── checkpoints/
    ├── epoch-1-half/
    ├── epoch-1/
    ├── epoch-2-half/
    └── epoch-2/
```

每个 milestone 保存 Adapter，不保存 optimizer resume state。若要真正支持中断续训，应另开
实验，把 optimizer/scheduler/RNG 一并 checkpoint，不能把当前 Adapter-only 目录当成完整
resume state。

### 8.5 监控什么

```powershell
nvidia-smi -l 2
```

关注：

- `train_loss` 和 `eval_loss` 是否有界；
- 验证 loss 是否下降但输出变短；
- 显存是否持续增长；
- 每半轮固定问题的回答；
- exact repeat、near repeat、空回答；
- 任务成功率、格式成功率、角色/领域一致性；
- 不知道、拒绝和安全边界是否退化。

遇到 NaN/Inf 时停止。遇到 OOM 时先保持 batch 1 和 gradient checkpointing，再降低
cutoff；若仍然 OOM，减少 target modules 或改用 QLoRA。

## 9. 如何选择 epoch 和最终检查点

不要用训练样本或验证 loss 单独选择最终模型。推荐流程：

1. Base 使用正式 system prompt 回答固定开发题；
2. 0.5/1/1.5/2 轮使用完全相同的 prompt、历史、seed 和采样参数回答；
3. 自动检查重复、长度、空回答、格式、关键词和硬失败；
4. 对回答来源做匿名 A/B；
5. 按预定义的硬门槛淘汰；
6. 在剩余候选中按自然度和任务质量选择；
7. 选定后只在封存 test 上运行一次。

硬门槛可以包括：

- 不泄露系统提示或用户隐私；
- 不将用户问题当成角色回答；
- 不编造工具结果、知识来源或现实行动；
- 重复率不高于 Base；
- 平均长度不出现灾难性坍缩；
- 用户明确结束时能够停止。

同一任务建议至少评价：

- 任务正确性；
- 指令遵循；
- 上下文承接；
- 格式稳定；
- 自然度/可读性；
- 长度适配；
- 重复；
- 事实/能力边界；
- 安全边界；
- 通用能力保持。

## 10. 固定问题评测

```powershell
python scripts\evaluate_checkpoints.py `
  --base-model models\qwen3-1.7b `
  --adapter-root outputs\lora-qwen3-1.7b-my-task `
  --system-prompt datasets\my-task\sft\system_prompt.txt `
  --output-dir outputs\my-task-evaluation `
  --max-new-tokens 192 `
  --input-budget 8192
```

脚本会识别：

```text
checkpoint-*
checkpoints/epoch-*
```

生成：

- `results.jsonl`：每次回答、Token、延迟和自动诊断；
- `summary.json`：各模型汇总；
- `report.md`：表格和原始回答。

内置问题只是通用烟测。正式任务应换成自己的固定开发集，并为格式、事实、拒答、工具调用或
其他硬约束增加程序检查。

## 11. 加载 Adapter 和 Base 对照

Base：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --system-prompt datasets\my-task\sft\system_prompt.txt `
  --max-new-tokens 256
```

LoRA：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter outputs\lora-qwen3-1.7b-my-task\checkpoints\epoch-1 `
  --system-prompt datasets\my-task\sft\system_prompt.txt `
  --max-new-tokens 256
```

固定对照时：

- 重启进程或 `/reset`；
- 使用同一基座 revision；
- system prompt 完全一致；
- 历史一致；
- temperature/top-p/top-k/repetition penalty 一致；
- thinking 设置一致。

Qwen3 官方模型卡建议 non-thinking 对话从 `temperature=0.7, top_p=0.8, top_k=20` 测试；
thinking 模式使用另一组参数。不要用 greedy decoding 证明模型复读，也不要为两个模型用
不同采样参数。

## 12. 与 LLaMA-Factory、TRL 的关系

本仓库的正式训练入口是 `train_lora.py`，因为 assistant-only mask 和检查点行为可以逐行检查。

如果更熟悉成熟框架：

- LLaMA-Factory 适合 QLoRA、多 GPU、可视化和大规模配置；
- `configs/*.yaml` 和 `dataset_info.json` 保留了兼容的 ShareGPT 字段；
- 但不同版本对 `mask_history`、历史 assistant loss、thinking 和保存策略的语义可能变化；
- 使用前必须按本节 dry-run 方法检查实际 labels，不能只因为“配置写了 mask_history”就默认正确；
- TRL 的 `SFTTrainer` 支持 conversational dataset 和 `assistant_only_loss=True`；
- 使用 TRL/LLaMA-Factory 会产生不同依赖和训练实现，不能与当前结果混为一组消融。

建议把框架版本、commit、完整 YAML、`dataset_info.json`、Token 统计和模板写入实验记录。

## 13. 常见故障

### `KeyError: 'qwen3'`

Transformers 太旧。使用支持 Qwen3 的版本，不要用 4.50 以下版本。

### CUDA 不可用

检查是否安装了 CPU PyTorch：

```powershell
python scripts\check_environment.py
```

不要设置 `CUDA_VISIBLE_DEVICES=-1` 后声称完成 GPU 复现。

### CUDA out of memory

顺序处理：

1. 关闭其他占显存进程；
2. 确认 batch 1；
3. 确认 gradient checkpointing；
4. 降低 cutoff；
5. 减少 target modules；
6. 使用 QLoRA 并另写实验记录。

### loss 下降但回答更差

这是正常可能。检查：

- 目标长度是否过度集中；
- 是否所有回答都使用同一模板；
- 学习率是否过高；
- epoch 是否过多；
- 是否只比较了训练 loss；
- 推理 prompt 是否和训练不同；
- decoder 参数是否导致复读。

### 验证 loss 很低但测试很差

可能发生了会话泄漏、近重复污染、测试题已经用于调参，或验证和测试分布不同。重新检查
会话 ID 与数据哈希，不要继续使用已污染的测试集证明效果。

### Adapter 加载报基座不匹配

确认 Adapter 与 `Qwen/Qwen3-1.7B` 相同 revision，且不是由 Qwen3-1.7B-Base、GPTQ 或
合并模型创建。`adapter_config.json` 只能记录原本训练的基座路径，不能修复结构或权重差异。

### 训练和推理回答风格不同

检查：

- system prompt 是否同一版本；
- 是否一个用 thinking、一个用 non-thinking；
- Qwen3 Chat Template 是否一致；
- 历史是否加了训练时没有的字段；
- 推理采样参数是否和评测一致。

## 14. 仓库内三个案例的数据来源

这一节只是通用流程的实际应用，不是训练模板的一部分。

| 数据集 | 角色来源 | 对话生成来源 | 可复现内容 |
|---|---|---|---|
| 许澄 | 项目原创 Character Bible | 网页 GPT 系列；60 个会话记录 `GPT-6`，70 个早期会话 `unknown` | 数据、训练配置、Adapter、评测 |
| 宋知意 | 项目原创角色卡 | OpenCode `deepseek-v4.1-flash` | 数据、训练配置、Adapter、评测 |
| 裴既明 | 项目原创角色圣经 | OpenCode `deepseek-v4.1-flash` | 数据、训练配置、Adapter、评测 |

三者均不含真实用户聊天。第三方角色实验的数据和权重受上游许可限制，不随仓库发布。

## 15. 参考依据

- [Qwen3-1.7B 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA-Factory 数据格式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)

更细的“采用了什么、没有采用什么”见 [`docs/REFERENCES.md`](REFERENCES.md)。
