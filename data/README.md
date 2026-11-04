# 数据说明

本仓库不附带 Chat-Huanhuan 的原始数据、衍生训练集或模型权重。

本项目原创角色许澄和宋知意的教师模型合成数据位于：

- `datasets/xu-cheng/`
- `datasets/song-zhiyi/`

早期 Chat-Huanhuan 实验曾使用项目公开的 `huanhuan.json`：

- 来源：<https://github.com/KMnO4-zx/huanhuan-chat>
- 训练文件：`dataset/train/lora/huanhuan.json`
- 数据类型：从《甄嬛传》剧本对白中整理的甄嬛问答对
- 官方行数：3,729

上游说明包含“仅供学习和研究”与“不得商用”等使用边界。项目中某个衍生模型卡使用了
不同的许可证声明，不能因此忽略底层剧本和角色素材涉及的第三方权利。

该实验现已判定为 **No-Go**：数据以单轮短问答为主，用户身份不固定，没有验证集，
持续聊天会明显退化为短句复读。仓库不提供训练配置、数据准备脚本或 Adapter；
失败原因见 [`experiments/huanhuan/README.md`](../experiments/huanhuan/README.md)。

使用前请自行复核：

1. 上游仓库当前的许可说明；
2. 底层文字和角色素材的使用权；
3. 是否允许公开衍生数据；
4. 是否允许发布或商业使用训练后的模型；
5. 是否需要在使用说明中署名原作者。

建议把下载文件放在：

```text
data/source/huanhuan.json
```

该目录已被 `.gitignore` 排除。

## ChatHaruhi / 凉宫春日

凉宫春日实验使用 ChatHaruhi 项目线路中的公开聚合对话：

- 代码与模型线路：<https://github.com/LC1332/Chat-Haruhi-Suzumiya>
- 聚合数据：`silk-road/ChatHaruhi-54K-Role-Playing-Dialogue`
- 固定 revision：`590c17dfc7d3f294bdae9c6aa9e6a7f0c753d487`
- 聚合数据中凉宫春日作为 assistant 的原始行数：1,163
- 清洗后训练 / 验证 / 独立评测：1,009 / 115 / 24

该数据卡与上游 README 对许可存在 **CC-BY-4.0 与 CC-BY-NC-4.0 的冲突声明**，
底层角色和作品素材还可能涉及其他第三方权利。因此仓库不附带原始数据、清洗后的衍生集、
Adapter 或量化权重，也不把本地非商业实验解释为可公开再分发授权。

实验参数、检查点和 No-Go 结论见
[`experiments/haruhi/README.md`](../experiments/haruhi/README.md)。
