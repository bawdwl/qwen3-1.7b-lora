# 数据说明

本仓库不附带 Chat-Huanhuan 的原始数据、衍生训练集或模型权重。

本项目原创角色许澄和宋知意的教师模型合成数据位于：

- `datasets/xu-cheng/`
- `datasets/song-zhiyi/`

参考实验使用 Chat-Huanhuan 项目公开的 `huanhuan.json`：

- 来源：<https://github.com/KMnO4-zx/huanhuan-chat>
- 训练文件：`dataset/train/lora/huanhuan.json`
- 数据类型：从《甄嬛传》剧本对白中整理的甄嬛问答对
- 官方行数：3,729

上游说明包含“仅供学习和研究”与“不得商用”等使用边界。项目中某个衍生模型卡使用了
不同的许可证声明，不能因此忽略底层剧本和角色素材涉及的第三方权利。

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
