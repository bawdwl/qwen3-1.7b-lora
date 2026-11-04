# Qwen3-1.7B LoRA：从环境到训练、评测与部署

这是一个可实际执行的 Qwen3-1.7B LoRA 训练仓库。`scripts/train_lora.py` 不绑定角色，
可以用于任意 system/user/assistant 对话数据；仓库内附三个原创角色实验，方便直接验证
数据、训练、评测和 Adapter 加载整条链路。

核心目标不是只展示几个模型结果，而是回答：

- 自己的数据能不能用于 Qwen3 LoRA；
- 训练集、开发集、测试集应该怎么切；
- assistant-only loss 如何实现，避免模型学习用户问题；
- rank、alpha、学习率、epoch、batch 和上下文长度如何选择；
- 怎么判断一个检查点是真的更好，而不是训练 loss 更低；
- 如何在 8 GB 显存机器上跑通并保留可比较的检查点。

**建议先读 [通用复现与参数指南](docs/REPRODUCTION.md)。** 本文提供最短路径和参数速查；
参考来源见 [官方文档与高星项目对照](docs/REFERENCES.md)。

## 1. 先确认应该训练哪种 Qwen3

| 场景 | 推荐模型 | 原因 |
|---|---|---|
| 中文角色聊天、陪伴、客服、问答 | `Qwen/Qwen3-1.7B` | 已经过预训练与后训练，具备 Chat Template 和指令跟随能力 |
| 只做普通聊天或低延迟应用 | `Qwen/Qwen3-1.7B` + `enable_thinking=False` | 避免生成额外思考文本 |
| 数学、代码、复杂推理，且数据包含思考过程 | `Qwen/Qwen3-1.7B` + thinking | 需要完整保留思考轨迹 |
| 从零学习语言/领域知识 | 通常不适合本仓库的小规模 LoRA | 基座预训练不是 SFT 的替代品 |
| 显存低于 6 GB | 使用 QLoRA 框架 | 本仓库内置脚本是 BF16 LoRA，目标机器为 8 GB 显存 |

本项目固定使用 `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。不要混用 Base、Instruct、GPTQ 或其他参数规模，
否则 Adapter 和实验结果不可直接比较。

## 2. 最短复现路径

### 2.1 安装环境

需要 Python 3.11、NVIDIA GPU、CUDA PyTorch。历史实验环境：

```text
RTX 4060 Laptop GPU 8 GB
Python 3.11.17
PyTorch 2.13.0+cu126
Transformers 5.20.2
PEFT 0.21.2
Accelerate 1.15.0
```

Windows PowerShell：

```powershell
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Linux：

```bash
git clone https://github.com/bawdwl/qwen3-1.7b-lora.git
cd qwen3-1.7b-lora
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### 2.2 下载模型

需要约 4.1 GB BF16 权重，放在 `models/qwen3-1.7b/`：

```bash
hf download Qwen/Qwen3-1.7B ^
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ^
  --local-dir models/qwen3-1.7b
```

PowerShell 使用反引号续行；Linux 用 `\`。也可以从 ModelScope 下载到同一目录。

### 2.3 检查机器和模型

```powershell
python scripts\check_environment.py `
  --model models\qwen3-1.7b
```

该脚本检查 Python、PyTorch、CUDA、BF16、显存、PEFT/Transformers 版本、模型权重分片、
Tokenizer 和 Qwen3 thinking/non-thinking Chat Template，不加载约 4.1 GB 模型权重到显存。

### 2.4 检查数据和标签

```powershell
python scripts\validate_datasets.py

python scripts\train_lora.py `
  --config configs\song_zhiyi_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b `
  --dry-run
```

dry-run 会 Tokenize 全部样本并报告输入 Token、监督 Token、历史裁剪、system prompt 哈希和
数据哈希。它还会验证 system/user/history 都被标记为 `-100`，只监督目标 assistant 和 EOS。

### 2.5 训练

仓库自带三个可直接运行的例子：

```powershell
# 276 个训练目标，推荐先从这个规模开始
python scripts\train_lora.py `
  --config configs\song_zhiyi_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b

# 576 个训练目标
python scripts\train_lora.py `
  --config configs\xu_cheng_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b

# 577 个训练目标 + 长回答权重
python scripts\train_lora.py `
  --config configs\pei_jiming_qwen3_1_7b_lora.yaml `
  --base-model models\qwen3-1.7b
```

未知数据从
[`configs/qwen3_1_7b_lora_template.yaml`](configs/qwen3_1_7b_lora_template.yaml)
复制一份，修改模型、数据目录、输出目录和参数即可。

### 2.6 评测与推理

```powershell
python scripts\evaluate_checkpoints.py `
  --base-model models\qwen3-1.7b `
  --adapter-root outputs\lora-song-zhiyi-v2 `
  --system-prompt datasets\song-zhiyi\sft\system_prompt.txt `
  --output-dir outputs\song-zhiyi-evaluation
```

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter outputs\lora-song-zhiyi-v2\checkpoints\epoch-2-half `
  --system-prompt datasets\song-zhiyi\sft\system_prompt.txt `
  --max-new-tokens 256
```

## 3. 数据应该怎么选和准备

LoRA 的目标是改变模型的表达方式、任务行为或特定领域能力，不适合让模型从零掌握大量新知识。
数据优先级通常是：

1. 自己拥有授权且贴近真实产品输入的数据；
2. 经过筛选的高质量公开对话或指令数据；
3. 教师模型按明确规范生成的合成数据；
4. 只有几条网页复制样本或高度重复的模板数据。

本项目提供的三个角色数据都属于第 3 类，不代表真实用户分布。

每条 SFT 样本至少应满足：

```json
{
  "system_prompt": "可选，但训练和推理应保持一致",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

处理步骤：

1. 删除空回复、损坏 JSON、控制 Token、重复模板和敏感个人信息；
2. 确认 user/assistant 交替，最后一条必须是目标 assistant；
3. 先在完整会话级划分 train/validation/test，再展开成 SFT；
4. 检查同一会话、相同问题或近似回答是否跨 split；
5. 统计长度、问句、拒答、动作和普通能力的分布；
6. 将 system、user 和旧历史 mask 掉，只监督最终 assistant 回答；
7. 保存固定哈希，训练后的失败样本不要删。

把完整多轮会话转换成 SFT：

```powershell
python scripts\build_sft.py `
  --input path\to\sessions.jsonl `
  --output path\to\sft\train.jsonl `
  --system-prompt path\to\system_prompt.txt
```

ShareGPT/LlamaFactory 风格的数据需要 `dataset_info.json`；本仓库三个数据目录都提供了
可直接运行的例子。完整字段、split 和清洗规则见
[复现指南的数据章节](docs/REPRODUCTION.md#5-选择与准备训练数据)。

## 4. Qwen3-1.7B LoRA 参数怎么选

先给出 8 GB 显存、单角色或单任务、约 300–2,000 个训练目标的保守起点：

| 参数 | 起点 | 推荐搜索范围 | 说明 |
|---|---:|---:|---|
| `lora_rank` | 8 | 4 / 8 / 16 | 4 更省参数，16 适合更复杂的风格或领域；先比较，不要盲升 |
| `lora_alpha` | 16 | 8 / 16 / 32 | 常从约 2×rank 开始；alpha 与 rank 共同决定缩放 |
| `lora_dropout` | 0.05 | 0–0.10 | 数据少且过拟合时可增加 |
| target modules | q/k/v/o + MLP | q/v；q/k/v/o；+gate/up/down | 全投影适配范围更大，显存和过拟合风险也更高 |
| `cutoff_len` | 1024 | 512–2048 | 先看 P95/P99 Token，通常留 15%–25% 余量 |
| micro batch | 1 | 1–4 | 受显存限制 |
| 梯度累积 | 8 | 4–16 | 目标有效 batch 通常约 8–16 |
| `learning_rate` | `2e-5` | `1e-5`–`1e-4` | 小数据优先低学习率；先用一个值跑半轮，不要同时改多项 |
| epoch | 1–2 | 0.5 / 1 / 1.5 / 2 | 必须比较检查点，不能只看 loss |
| scheduler | cosine | linear/cosine | 小数据常用 cosine |
| warmup | 5% | 3%–5% | 避免最初几步更新过猛 |
| weight decay | 0.01 | 0–0.05 | 常见保守值 |
| precision | BF16 | BF16/FP16 | 支持时应优先 BF16 |
| gradient checkpointing | true | true/false | 8 GB 显存建议开启 |

### 4.1 rank 怎么理解

rank 是 LoRA 更新矩阵的秩。提高 rank 会增加可训练参数和表达能力：

- 小数据、只改语气或格式：先试 rank 4 或 8；
- 单角色、稳定风格、少量知识：rank 8 是常用起点；
- 域复杂或明显欠拟合：再试 16；
- rank 32 以上通常需要更可靠的数据和更严格的评测，不会自动优于 8。

本项目裴既明第一版使用 rank 8、七个投影层时回答快速变短；降低为 rank 4、只训练
`q/k/v/o` 后有所缓解，但仍是具体数据上的取舍，不是“rank 越小越好”的通用规律。

### 4.2 target modules 怎么理解

- `q_proj, v_proj`：最小、最省显存；
- `q/k/v/o`：标准注意力层适配，8 GB 显存优先从这里开始；
- 再加 `gate/up/down`：同时适配 MLP，通常表达能力更强，也更容易放大训练数据中的短句和模板。

### 4.3 学习率怎么选

没有适用于所有数据的单一学习率。建议先固定数据、batch 和 epoch，只做一次学习率对照：

| 数据情况 | 初始学习率 |
|---|---:|
| 少于约 500 个目标，已使用较强系统提示 | `1e-5`–`2e-5` |
| 500–2,000 个目标 | `2e-5`–`5e-5` |
| 大于 5,000 个目标 | `5e-5`–`1e-4` |

若回答迅速变成固定短句、空回答或只复述用户问题，优先降低学习率或减少 epoch。

### 4.4 epoch 怎么选

不要根据“一般训练 3 轮”拍脑袋。先计算优化步数：

```text
每轮优化步数
= ceil(训练样本数 / (micro_batch × gradient_accumulation × GPU 数))

总步数 = 每轮步数 × epoch
```

例如：

```text
276 个样本，batch 1，梯度累积 8，1 张 GPU
每轮 = ceil(276 / 8) = 35 步
2 轮 = 70 步
```

推荐保存 0.5、1、1.5、2 轮，并在开发集使用同一提示和采样参数比较。经验起点：

- 少于 500 个目标：先比较 0.5、1、1.5 轮；
- 500–2,000 个目标：先比较 1、1.5、2 轮；
- 大于 5,000 个目标：先跑 1 轮，确认方向后再决定是否继续；
- 出现复读、长度坍塌、身份混淆或边界退化：停止增加 epoch。

验证 loss 下降只说明更会预测验证目标 Token，不等于用户需要的回答更好。

## 5. 训练过程

训练脚本会冻结基座、注入 LoRA、应用 Qwen3 Chat Template，并只对最终 assistant 输出和 EOS 计算
损失。超长样本会先移除最早的完整历史轮次，绝不会静默截断监督目标。

训练中至少观察：

- train/eval loss 是否有限；
- grad norm 是否长期异常；
- 显存是否稳定；
- 半轮和整轮回答是否变短、复读或身份混乱；
- 平均长度、问句比例和安全边界是否明显偏离 Base。

常见问题：

- 不是改号，而是先运行 `check_environment.py` 检查环境；
- OOM 时先确认 batch=1、gradient checkpointing 开启，再降低 cutoff；
- 回答重复时同时检查数据长度分布、学习率、epoch 和生成参数；
- 训练、评测和部署的提示词、thinking 模式、模板版本必须一致。

完整训练监控、选点和故障排查见
[复现指南](docs/REPRODUCTION.md#8-开始训练并监控)。

## 6. 仓库结构

```text
configs/
  qwen3_1_7b_lora_template.yaml   通用参数模板
  song_zhiyi...yaml               276 目标示例
  xu_cheng...yaml                 576 目标示例
  pei_jiming...yaml               577 目标 + 长度权重示例
datasets/                         三个完整可复现示例及其来源
scripts/
  check_environment.py            环境/显存/模型检查
  build_sft.py                    多轮会话展开为 SFT
  train_lora.py                   通用 Qwen3-1.7B LoRA 训练
  evaluate_checkpoints.py         Base/检查点固定题集对照
  chat_checkpoint.py              Base/Adapter 终端推理
  validate_datasets.py            结构、split、哈希校验
src/lora_chat.py                  PEFT 推理封装
weights/                          已发布 Adapter 与哈希
docs/                             复现、参数、参考、结果
```

## 7. 复现边界

- 实测过 RTX 4060 Laptop 8 GB、BF16、单卡训练；
- 脚本使用 Transformers + PEFT，不依赖 LLaMA-Factory；
- LLaMA-Factory、TRL、QLoRA 的对应关系见
  [参考资料](docs/REFERENCES.md)；
- 本项目不是论文级最优超参数搜索；
- 不包含 Qwen3 基座权重、API Key 或真实用户聊天；
- 不同 CUDA/PyTorch/GPU 环境不承诺逐 bit 一致；
- 所有发布 Adapter 都必须只用开发集选点，封存测试只运行一次。

## 8. 项目内案例与验证

下列内容是通用流程的应用记录，应放在复现文档之后阅读：

- [数据来源总表](data/README.md)
- [许澄数据](datasets/xu-cheng/README.md)
- [宋知意数据](datasets/song-zhiyi/README.md)
- [裴既明数据](datasets/pei-jiming/README.md)
- [三个已发布 Adapter](weights/README.md)
- [训练结果与失败模式](docs/FINETUNING_CASE_STUDY.md)
- [Base/LoRA 同条件推理](docs/LOCAL_COMPARISON.md)
- [全部 Base/LoRA 对话截图](screenshots/README.md)

数据来源明确区分：

| 角色 | 来源 | 数据与权重 |
|---|---|---|
| 许澄 | 原创 Character Bible；网页 GPT 系列，60 个会话记录 `GPT-6`、70 个早期会话 `unknown` | 数据和 Adapter 均已发布 |
| 宋知意 | 原创角色卡；OpenCode `deepseek-v4.1-flash` 生成 | 数据和 Adapter 均已发布 |
| 裴既明 | 原创角色圣经；OpenCode `deepseek-v4.1-flash` 生成 | 数据和 Adapter 均已发布 |
| 孙悟空 | RoleBench 衍生数据 | 因第三方衍生边界不发布 |
| 凉宫春日 | ChatHaruhi 固定子集 | 因上游许可声明冲突不发布 |

### ModelScope MNN Q8 模型

三个原创角色的 LoRA 已合并到 Qwen3-1.7B，并以 MNN 3.5.0 Q8 block64 发布到 ModelScope：

| 角色 | 检查点 | MNN 模型 |
|---|---|---|
| 宋知意 | 1.5 轮 | <https://modelscope.cn/models/Alixs666/shiyu-qwen3-1.7b-song-zhiyi-mnn-q8> |
| 许澄 | 第 1 轮 | <https://modelscope.cn/models/Alixs666/shiyu-qwen3-1.7b-xu-cheng-mnn-q8> |
| 裴既明 | 第 1 轮 | <https://modelscope.cn/models/Alixs666/shiyu-qwen3-1.7b-pei-jiming-mnn-q8> |

这些模型包面向 Android/MNN Chat 的本地文字推理，包含角色提示词、导出参数和文件哈希。
Q8 是 MNN 权重量化格式，不是 GPTQ。许澄和裴既明是实验版，不代表生产级安全通过；
端侧性能仍需在目标手机上复测。

截图只是单次实际对话，不是统计证明。训练 loss、自动指标和单一 LLM 裁判都不能替代
固定开发集、封存测试集和人工盲评。

## 9. 验证仓库

```bash
python scripts/check_environment.py --model models/qwen3-1.7b
python scripts/validate_datasets.py
python -m compileall -q src scripts
python scripts/check_markdown_links.py
```

GitHub Actions 会检查数据哈希、结构、Python 语法和 Markdown 链接。完整模型训练必须由使用者
在本地 GPU 环境执行，CI 不下载 Qwen3 基座。
