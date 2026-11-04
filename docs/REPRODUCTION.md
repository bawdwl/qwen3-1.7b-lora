# 从零复现 Qwen3-1.7B 单角色 LoRA

本文是仓库的正式复现指南。目标是让一个没有本项目历史文件、聊天数据库或旧虚拟环境的
人，能够：

1. 安装兼容的 Python 环境；
2. 下载固定版本的 Qwen3-1.7B；
3. 检查训练数据结构和 split；
4. 实际执行 assistant-only LoRA 训练；
5. 观察 loss、显存、长度和复读风险；
6. 比较 Base、半轮、整轮和后续检查点；
7. 使用仓库中的 Adapter 做推理；
8. 判断哪些结果可以复现，哪些结论只适用于本项目的实验设置。

它描述的是小规模单角色实验，不是多角色产品训练平台，也不是论文级复现。

## 1. 先理解仓库里有什么

### 1.1 代码

- `scripts/train_lora.py`：仓库内自包含的训练入口；
- `scripts/chat_checkpoint.py`：加载 Base 或 PEFT Adapter 进行终端对话；
- `scripts/evaluate_checkpoints.py`：固定问题集上的 Base/检查点对照；
- `scripts/validate_datasets.py`：数据行数、格式、split 和 SHA-256 检查；
- `src/lora_chat.py`：Transformers + PEFT 推理封装。

训练不依赖私有服务或聊天 App。数据、训练配方和推理入口都在本仓库中。

### 1.2 可完整复现的实验

| 角色 | 数据已包含 | 发布 Adapter | 可复现范围 |
|---|---|---|---|
| 宋知意 | 是 | 是 | 数据检查、训练、Base/LoRA 对照、已有 Adapter 推理 |
| 许澄 | 是 | 是 | 数据检查、训练、Base/LoRA 对照、已有 Adapter 推理 |
| 裴既明 | 是 | 是 | 数据检查、训练、Base/LoRA 对照、已有 Adapter 推理 |
| 孙悟空 | 否 | 否 | 只保留配置说明和实际对话截图，第三方衍生数据不公开 |
| 凉宫春日 | 否 | 否 | 只保留来源、许可冲突和实验记录，第三方数据和权重不公开 |

“完整复现”表示命令和所需数据都在仓库中。由于 CUDA、PyTorch、GPU 型号和并行环境可能
不同，不承诺与历史训练逐 bit 相同。

### 1.3 数据来源

许澄、宋知意、裴既明都是项目原创虚构角色。全部训练对白均由教师大语言模型根据项目
作者定义的角色设定、场景规范和审核规则生成。它们不包含真实用户聊天、真实社交平台
抓取内容或原小说对白。

具体来源：

| 数据集 | 角色来源 | 对话来源 | 冻结规模 |
|---|---|---|---|
| 许澄 | 项目原创 Character Bible | 网页 GPT 系列；60 个会话记录 `GPT-6`，70 个早期会话 `unknown` | 80/20/30 个会话；576/142 个训练/验证目标 |
| 宋知意 | 项目原创角色卡 | OpenCode `deepseek-v4.1-flash` 按 80 个校园场景生成 | 64/8/8 个会话；276/35/35 个目标 |
| 裴既明 | 项目原创乙女向角色圣经 | OpenCode `deepseek-v4.1-flash` 按 120 个场景生成，加 60 条长回复分支 | 96/13/11 个场景；577/54/45 个目标 |

数据中保留 teacher prompt、角色设定和审计文件，目的是让使用者能判断数据是怎样产生的，
也能按同样规范制作自己的角色数据。

第三方实验边界：

- 孙悟空使用过 RoleBench 衍生数据，但当前仓库不重新分发数据或 Adapter；
- 凉宫春日使用过 ChatHaruhi 54K 聚合数据中的固定子集，但上游数据许可说明冲突，
  当前仓库不重新分发数据或 Adapter；
- 截图只能说明作者实际体验过一次对话，不是统计实验结果。

## 2. 硬件和软件

历史训练在一台 `RTX 4060 Laptop GPU 8 GB` 上完成。推荐最低配置：

- NVIDIA GPU，8 GB 显存；
- 16 GB 系统内存，建议 32 GB；
- 10 GB 以上空闲磁盘用于 PyTorch、Qwen3 和 Adapter；
- Windows 10/11 或 Linux；
- Python 3.11；
- 支持 BF16 的 CUDA 环境。

历史 Adapter 元数据记录的软件版本：

```text
Python       3.11.17
PyTorch      2.13.0+cu126
Transformers 5.20.2
PEFT         0.21.2
Accelerate   1.15.0
```

`requirements.txt` 固定了这些非 PyTorch 版本。PyTorch 需要按操作系统和 CUDA 版本从
官方源安装。安装后先验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

输出必须是 `True` 并显示 NVIDIA GPU。`False` 时停止训练，不要用 CPU 跑通后声称复现了
历史实验。

## 3. 从干净目录开始

克隆仓库并创建隔离环境：

```powershell
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux：

```bash
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

安装 CUDA PyTorch。以下示例匹配历史 CUDA 12.6 环境；如果本机需要其他 CUDA，请使用
PyTorch 官网对应命令：

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

再安装其他固定依赖：

```bash
python -m pip install -r requirements.txt
```

若已经有兼容 PyTorch，不要让 `pip` 再安装一个不同 CUDA 构建覆盖它；先核对版本。

## 4. 下载固定版本基座模型

训练脚本默认从下面的位置读取模型：

```text
models/qwen3-1.7b/
```

该目录在 `.gitignore` 中，模型权重不会进入 Git。

Hugging Face 方式：

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen3-1.7B ^
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ^
  --local-dir models/qwen3-1.7b
```

Linux 将行尾的 `^` 改成 `\`。如果当前 `huggingface-cli` 版本不支持 `hf` 子命令，可使用：

```bash
huggingface-cli download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/qwen3-1.7b
```

ModelScope 方式：

```bash
python -m pip install modelscope
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/qwen3-1.7b
```

下载后至少应有：

```text
config.json
generation_config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

本项目使用的 Qwen3-1.7B BF16 文件合计约 4.1 GB。不要拿 Qwen3-0.6B、GPTQ 版本或其他
revision 的结果冒充相同复现。

## 5. 先验证数据，不加载模型

运行仓库级数据检查：

```bash
python scripts/validate_datasets.py
```

它会检查：

- 所有必需 JSONL/JSON 文件存在；
- 行数是否与冻结版本一致；
- SFT 最后一条消息是否为非空 assistant；
- 数据集 manifest 中的 SHA-256 是否与实际文件一致；
- `dataset_info.json` 是否指向正确文件。

最强的检查失败时不要继续训练。查看 `datasets/manifest.json` 可以定位文件差异。

## 6. 数据结构与 split

### 6.1 source 和 SFT 的区别

`source/` 保存一段完整多轮对话。例如一段对话包含 4 个 user/assistant 回合。

`sft/` 将这段对话展开成 4 个训练样本：

1. 只给第 1 个 user，目标第 1 个 assistant；
2. 给第 1 个 user、第 1 个 assistant、第 2 个 user，目标第 2 个 assistant；
3. 给更长历史，目标第 3 个 assistant；
4. 给完整历史，目标第 4 个 assistant。

这样可以学习“在已有上下文下生成当前回复”，不是让模型一次背完整段对话。

仓库中的冻结 SFT 可以直接训练。如果要把自己的多轮会话转换成相同格式，可运行：

```powershell
python scripts\build_sft.py `
  --input path\to\sessions.jsonl `
  --output path\to\sft\train.jsonl `
  --system-prompt path\to\system_prompt.txt
```

JSON 输入也可以指定列表键：

```powershell
python scripts\build_sft.py `
  --input path\to\dialogues.json `
  --input-key dialogues `
  --output path\to\sft\train.jsonl `
  --system-prompt path\to\system_prompt.txt
```

脚本要求每个会话以 user 开始、user/assistant 严格交替、以 assistant 结束；每遇到一个
assistant 就生成一个目标，并保留会话 ID 等来源字段。它不会替你清洗内容或重新切分
split，转换前仍要完成上一节的数据审查。

### 6.2 为什么必须按会话拆分

同一段对话展开出来的多个目标高度相关。如果先把 4 个样本中的 3 个放进 train、1 个放进
validation，验证集就泄漏了训练上下文，loss 会显得过于乐观。

因此仓库先按原始会话冻结 split，再分别展开 SFT。不要随机打乱所有 JSONL 行后重新切分。

封存测试集在检查点选择完成前不得查看。若看过测试结果后再调参，同批测试就不再是封存测试。

### 6.3 训练提示与推理提示

每条 SFT 都保留自己的 `system_prompt`。训练和推理应尽量使用同一版本：

- 宋知意：`datasets/song-zhiyi/sft/system_prompt.txt`
- 许澄：`weights/xu-cheng-epoch-1/system_prompt.txt`
- 裴既明：`datasets/pei-jiming/sft/system_prompt.txt`

训练时写完整角色卡，部署时换成很短的核心提示，会改变条件分布。它可以作为一个实验，
但不能把混合提示的结果归功于模型权重。

## 7. assistant-only 标签规则

`scripts/train_lora.py` 使用以下算法：

1. 把 system prompt 和最终 assistant 之前的所有消息套用 Qwen3 Chat Template；
2. 追加当前 assistant 目标文本和 EOS；
3. Tokenize 后先验证追加文本没有改变已有 Token 前缀；
4. system、user、历史 assistant 的位置全部设为 `-100`；
5. 仅最终 assistant 回答和 EOS 保留真实 Token ID；
6. 超长时只移除最早的完整 user/assistant 历史对，绝不静默截断监督目标；
7. 如果 system、最终 user 和目标本身仍超过最大长度，直接报错。

训练前运行 dry-run：

```powershell
python scripts\train_lora.py `
  --config configs\song_zhiyi_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b `
  --dry-run
```

Linux 将反引号替换成 `\`。输出会给出：

- train/validation 行数；
- 输入 Token 的 min/P50/P90/max；
- 监督 Token 的 min/P50/P90/max；
- 因超长移除的历史对数；
- system prompt 和数据的 SHA-256；
- 标签策略和样本权重设置。

常见错误：

- 用户消息也被监督：角色会学习生成用户的话；
- 整个历史都被监督：模型会记忆上下文，而不是只学习当前回复；
- 先截断再做 mask：可能最终目标全部变成 `-100`；
- `DataCollatorForLanguageModeling` 覆盖自定义 labels；
- 用字符串字符数替代 Token 数；
- 目标和 EOS 没有一起监督，导致模型不会自然结束。

## 8. 数据清洗与设计检查

本仓库的数据已经冻结，下面列出重新制作数据时必须检查的项目。

### 8.1 格式和角色

- 每段对话从 user 开始，以 assistant 结束；
- user/assistant 必须交替；
- 删除空回复、重复 JSON 字段和损坏 Unicode；
- 删除 `<|im_start|>`、`<|im_end|>` 等模板控制 Token 的原始文本；
- system prompt 不能在每条对话中发生无记录漂移；
- 每条样本的来源 ID、场景 ID 和 split 必须可追踪。

### 8.2 内容质量

- 角色名称、年龄、职业和关系不能前后矛盾；
- 不能把用户的未知经历写成角色已知事实；
- 同一会话可以有即兴细节，但必须前后一致；
- 删除“作为 AI”“希望这些建议有帮助”等客服/助手腔；
- 删除固定安慰模板和每轮反问；
- 保留自然短句，避免所有回答都变成 80 字小作文；
- 训练数据中的长度、动作和问句比例应符合部署目标。

### 8.3 去重和泄漏

- 同一个 assistant 目标不要机械复制到多个场景；
- 计算规范化后的完全重复和近重复；
- train、validation、test 间检查用户首句、场景和角色回复；
- 同一 `session_id` 不得跨 split；
- 测试集不能参与纠错、提示词选择或训练轮数选择。

### 8.4 隐私和许可

- 不使用真实用户聊天，除非已获得明确授权并完成脱敏；
- 不使用 API Key、手机号、地址、账号凭证等个人信息；
- 第三方 IP 数据必须核对底层作品、数据集卡和衍生模型许可；
- 数据审查通过不等于自动获得商业再分发权。

## 9. 训练参数及选择理由

仓库里的参数是为 1.7B、单角色、小到中等规模数据设计的起点，不是所有角色的统一最优值。

### 9.1 许澄

```text
LoRA rank        8
LoRA alpha       16
LoRA dropout     0.05
Target modules   q/k/v/o/gate/up/down
Epochs           2
LR               5e-5
Cutoff           1024
Batch            1
Accumulation     8
Scheduler        cosine
Warmup           0.05
```

理由：576 个训练目标比宋知意、裴既明更多，七个投影层可以观察单角色风格适配；每轮约
72 个优化步。

### 9.2 宋知意

```text
LoRA rank        8
LoRA alpha       16
LoRA dropout     0.05
Target modules   q/k/v/o/gate/up/down
Epochs           2
LR               2e-5
Cutoff           1024
Batch            1
Accumulation     8
Scheduler        cosine
Warmup           0.05
```

理由：只有 276 个训练目标，降低学习率以减缓快速模式坍塌；仍然保留半轮和整轮比较。

### 9.3 裴既明

```text
LoRA rank        4
LoRA alpha       8
LoRA dropout     0.05
Target modules   q/k/v/o
Epochs           2
LR               2e-5
Cutoff           768
Batch            1
Accumulation     8
Scheduler        cosine
Warmup           0.04
Long target      60+ Chinese chars: 3x sample weight
```

理由：第一轮 rank 8 + 七投影层实验出现快速长度收缩，因此最终配方减少可训练容量，
并提高长目标权重。这个处理改善了一些开发集指标，但没有消除 LoRA 比 Base 更短的问题。

### 9.4 选择逻辑

- 先跑 Base + 正式提示，不把训练当作第一步；
- 数据少于约 1,000 个目标时，优先比较 0.5、1、1.5、2 轮；
- 不因 loss 继续下降就默认训练更久；
- 不以同一角色的训练样本代替验证；
- 检查点选择在 validation/development 上完成；
- 最终只用一次封存 test。

## 10. 开始训练

训练前删除或改名默认输出目录，脚本拒绝写入非空目录，避免覆盖旧结果。

宋知意：

```powershell
python scripts\train_lora.py `
  --config configs\song_zhiyi_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b
```

许澄：

```powershell
python scripts\train_lora.py `
  --config configs\xu_cheng_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b
```

裴既明：

```powershell
python scripts\train_lora.py `
  --config configs\pei_jiming_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b
```

Linux 将反引号替换成 `\`。

首次运行时 Transformers 需要加载约 4.1 GB BF16 权重。历史 8 GB 显卡实验的峰值已分配
显存约 4.3–4.6 GiB，但不同 PyTorch/CUDA 版本可能更高。

训练输出示意：

```text
outputs/lora-song-zhiyi-v2/
├── training-audit-before.json
├── training-audit.json
├── tensorboard/
├── adapter_config.json
├── adapter_model.safetensors
└── checkpoints/
    ├── epoch-1-half/
    ├── epoch-1/
    ├── epoch-2-half/
    └── epoch-2/
```

每个 milestone 只保存 Adapter，不保存完整优化器，因此可以比较模型效果，但不能从中断步
Exactly resume optimizer 状态。

## 11. 训练时看什么

### 11.1 loss

- loss 必须是有限值；
- 前几十步通常波动，不需要因为单步反弹停止；
- 验证 loss 快速接近 0 但回答越来越短，可能是记住高频短句式；
- 验证 loss 下降但自由对话变差，应视为分布和评估目标不一致。

### 11.2 显存

```powershell
nvidia-smi -l 2
```

若 8 GB 显卡 OOM：

1. 确认只训练 LoRA，冻结基座；
2. 确认 `gradient_checkpointing: true`；
3. 保持 batch 1，只降低 cutoff；
4. 仍不够时减少 target modules 或 rank；
5. 不要用“悄悄切换到其他基座”掩盖 OOM。

### 11.3 质量监控

每个半轮检查点都做 10–20 个固定聊天，观察：

- 是否开始重复上一轮回答；
- 是否只会回答几个字；
- 是否忘记角色名或把用户身份说错；
- 是否每轮反问；
- 是否把动作写成固定模板；
- 是否在不知道时编造；
- 是否比同提示 Base 更自然、更连贯。

出现稳定复读、身份混乱、极短回答或边界恶化时，先停止增加 epoch，不要用降低 loss 解释。

## 12. 检查点选择

同一角色的 Base 与所有检查点必须使用：

- 同一基座 revision；
- 同一 system prompt；
- 同一聊天历史；
- 同一生成参数；
- 同一最大输出长度；
- 同一评估问题。

先抽查开发集，再计算自动诊断：

- 完全重复回答；
- 相邻回答近似复读；
- 过短回答；
- 平均/中位字符数；
- 角色称谓和身份混乱；
- 事实冲突；
- 安全边界；
- 多轮上下文承接；
- 人工 A/B 偏好。

验证 loss 只用于发现训练异常，不能单独决定“哪个模型更像角色”。

## 13. 固定问题的 Base/LoRA 对照

运行：

```powershell
python scripts\evaluate_checkpoints.py `
  --base-model models\qwen3-1.7b `
  --adapter-root outputs\lora-song-zhiyi-v2 `
  --system-prompt datasets\song-zhiyi\sft\system_prompt.txt `
  --output-dir outputs\song-zhiyi-evaluation
```

脚本会识别两类目录：

```text
checkpoint-*
checkpoints/epoch-*
```

如角色需要称谓统计，可重复传入：

```powershell
python scripts\evaluate_checkpoints.py ... --role-marker 宋知意
```

输出：

- `results.jsonl`：每个问题的原始回答、Token、延迟和自动诊断；
- `summary.json`：各检查点的机器可读汇总；
- `report.md`：便于人工查看的表格和回答原文。

自动指标不能识别所有语义错误。最终仍需要人工或独立裁判，最好对模型来源匿名。

## 14. 使用已经发布的 Adapter

Base 对照组：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256
```

LoRA 组：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter weights\song-zhiyi-epoch-2-half `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256
```

其他角色替换 Adapter 和提示词目录：

```text
weights/xu-cheng-epoch-1/
weights/pei-jiming-epoch-1/
```

固定比较时，每次输入 `/reset` 清空历史。不要把 LoRA 的历史继续喂给 Base，或反过来。

## 15. 常见故障

### `CUDA is required`

当前 PyTorch 是 CPU 构建，或 NVIDIA 驱动/CUDA 不可用。重新安装匹配机器的 CUDA wheel。

### CUDA OOM

确认没有旧 Python 进程占用显存：

```powershell
nvidia-smi
```

关闭占用进程后重试。不要在没有记录的情况下改 cutoff 或 LoRA target；这会产生不同实验。

### `Adapter was trained for ...`

Adapter 与基座模型不一致。使用 `Qwen/Qwen3-1.7B` 固定 revision，而不是 0.6B、4B 或另一个
量化基座。

### 找不到 `adapter_model.safetensors`

确认路径指向 Adapter 目录本身，不是仓库根目录；PEFT 需要
`adapter_config.json` 和权重文件同时存在。

### 回答为空或不停重复

先检查：

- 训练和推理提示是否一致；
- 检查点是否过晚；
- 训练目标是否过短；
- 历史中是否已经包含模型生成的重复错误；
- `max_new_tokens`、temperature 和 repetition penalty 是否一致。

复读通常是数据分布、检查点和上下文的共同问题，不是简单加一句“不要重复”就能修复。

### Windows `KMP_DUPLICATE_LIB_OK`

这是危险提示，表示同一进程加载了多份 OpenMP runtime。优先修复环境冲突。不得把
`KMP_DUPLICATE_LIB_OK=TRUE` 当成默认解决方案；它可能隐藏崩溃或给出错误结果。

### 本地 API Key 无法使用

本仓库训练和推理直接加载本地权重，不需要 DeepSeek、OpenCode 或 Qwen API Key。本地假
Key、`.env` 和 App 配置不会影响训练。真正的模型下载认证只由 Hugging Face/ModelScope
工具处理。

## 16. 复现结论的边界

- 训练成功只证明流程可执行；
- loss 下降不证明角色质量提高；
- 单张截图不是统计评测；
- 单一 LLM 裁判不是人工盲评；
- 276–577 个训练目标不能代表真实用户分布；
- 1.7B 单角色 Adapter 不能替代角色卡、上下文管理、长期记忆和安全策略；
- 已发布 Adapter 是作者偏好的实验候选，不是生产安全认证；
- 看过 test 后继续调参，就不能再把同一 test 称为封存测试。

如果这些限制得到如实报告，本仓库就完成了它要证明的事情：从数据、标签、训练、选点、
推理到错误分析，整条单角色 LoRA 流程可以被检查，也可以在其他人的机器上重新执行。
