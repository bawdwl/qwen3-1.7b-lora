# Base / LoRA 实际对话截图

以下截图由作者在本机终端实际对话生成。文件名 `base.png` 表示未加载 LoRA，
`lora.png` 表示加载对应角色 Adapter。

| 角色 | 使用的 LoRA 配置 | Base / LoRA |
|---|---|---|
| 孙悟空 | 第一版 `checkpoint-140` + 核心提示 | [base](wukong/base.png) / [lora](wukong/lora.png) |
| 许澄 | `epoch-1` + 核心提示 | [base](xu-cheng/base.png) / [lora](xu-cheng/lora.png) |
| 宋知意 | `epoch-2-half`（1.5 轮）+ 精简提示 | [base](song-zhiyi/base.png) / [lora](song-zhiyi/lora.png) |
| 凉宫春日 | `checkpoint-189`（1.5 轮）+ 完整角色卡 | [base](haruhi/base.png) / [lora](haruhi/lora.png) |
| 裴既明 | `epoch-1` + 精简提示 | [base](pei-jiming/base.png) / [lora](pei-jiming/lora.png) |

## 统一阅读说明

- 截图展示的是作者实际体验中的一小段对话，不是完整统计评测。
- 截图中保留了命令行生成参数和耗时，便于确认两边不是随手拼接的结果。
- 某些较晚检查点和提示组合已经知道会产生过短、复读或边界问题，
  因此不能仅凭一张更顺眼的截图说明模型整体质量通过。
- 不提供 API Key、账号信息、聊天数据库或个人录音。
