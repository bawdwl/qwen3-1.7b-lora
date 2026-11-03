# Base / LoRA 对话截图

这里用于放作者实际对话截图。每张截图应同时保留角色提示版本和模型模式，
或者使用以下目录与文件名记录：

```text
screenshots/
├── song-zhiyi/
│   ├── base-01.png
│   ├── lora-01.png
│   └── notes.md
└── xu-cheng/
    ├── base-01.png
    ├── lora-01.png
    └── notes.md
```

`notes.md` 应写明：

- 基座模型版本；
- Adapter 检查点；
- 使用的 System Prompt 文件；
- 提问顺序；
- 采样参数；
- 截图是否来自同一次启动、是否清空了历史。

截图是单次体验证据。不要只挑选成功案例；如果 LoRA 变短、复读、答非所问
或违反边界，也应保留对应截图。
