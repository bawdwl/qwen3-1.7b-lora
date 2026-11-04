# 数据说明

本项目原创角色许澄、宋知意和裴既明的教师模型合成数据位于：

- `datasets/xu-cheng/`
- `datasets/song-zhiyi/`
- `datasets/pei-jiming/`

三个角色均为项目原创虚构人物。对话由教师大语言模型根据作者定义的角色设定、场景规范
和审核规则生成，不包含真实用户聊天记录。数据来源、生成方式、split 和限制分别记录在
各数据集目录中。

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

实验参数、检查点和质量结论见
[`experiments/haruhi/README.md`](../experiments/haruhi/README.md)。
