# 本地 Base / LoRA 对照

## 对照原则

每次比较必须固定以下条件：

1. 同一个 `Qwen/Qwen3-1.7B` 基座权重和 revision；
2. 同样的角色提示词文件；
3. 从空对话开始；
4. 相同的用户问题顺序；
5. 相同的 `temperature`、`top_p`、`top_k`、重复惩罚和输出上限。

先在 Base 中完整回答一组问题，再重启进程加载 LoRA，避免上一模型的回答
通过历史上下文污染下一次比较。

## 宋知意

Base：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256 `
  --temperature 0.7 `
  --top-p 0.8 `
  --top-k 20 `
  --repetition-penalty 1.0
```

LoRA：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter weights\song-zhiyi-epoch-2-half `
  --system-prompt weights\song-zhiyi-epoch-2-half\system_prompt.txt `
  --max-new-tokens 256 `
  --temperature 0.7 `
  --top-p 0.8 `
  --top-k 20 `
  --repetition-penalty 1.0
```

## 许澄

Base：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --system-prompt weights\xu-cheng-epoch-1\system_prompt.txt `
  --max-new-tokens 256 `
  --temperature 0.7 `
  --top-p 0.8 `
  --top-k 20 `
  --repetition-penalty 1.0
```

LoRA：

```powershell
python scripts\chat_checkpoint.py `
  --base-model models\qwen3-1.7b `
  --adapter weights\xu-cheng-epoch-1 `
  --system-prompt weights\xu-cheng-epoch-1\system_prompt.txt `
  --max-new-tokens 256 `
  --temperature 0.7 `
  --top-p 0.8 `
  --top-k 20 `
  --repetition-penalty 1.0
```

## 建议截图

截图至少覆盖：

- 一个日常闲聊问题；
- 一个需要承接前文的多轮问题；
- 一个角色性格或关系问题；
- 一个角色不应知道或不应照做的边界问题。

建议文件名：

```text
screenshots/song-zhiyi/base-01.png
screenshots/song-zhiyi/lora-01.png
screenshots/xu-cheng/base-01.png
screenshots/xu-cheng/lora-01.png
```

截图只能证明一次实际对话，不替代固定题集统计。描述时应注明这是单次
对话观察，并保留不理想的回答。
