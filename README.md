# AtomGPT （正在编辑中）
为了能够在中文上训练出一个能够和ChatGPT能力接近的中文大模型，我们开放了AtomGPT项目，AtomGPT基于LLaMA的模型架构，从0开始训练，希望能在训练的过程中，将模型能力得到提升的进化过程展示出来，感受到模型学习的过程。
## 模型在线体验平台
能够更加直观可视化出模型训练的过程中模型能力的变化，以及方便进行测试，我们搭建了AtomGPT模型成长平台。

在该平台上，我们提供了一个在线测试入口，点击右上角体验一下，注册登录即可体验（ps:当前显卡资源有限，有时会出现排队的情况）

体验地址：[https://grow.atomecho.cn/](https://grow.atomecho.cn/)

## 最新更新

### 新闻动态

### 预训练模型更新
- 2023.06.01 


### chat模型更新
- 2023.06.01 


## 训练细节
我们基于transformers实现的LLaMA模型代码，参考meta开源的13B的模型配置，作为实现AtomGPT的开始。训练过程中使用了10台8卡A100的机器，在bf16的精度上，进行了全量参数微调。
数据来源方面，主要包含以下几方面的数据。
1. 中文数据
中文数据作为了预训练的主要数据部分，主要来源有以下几个部分
- 由原子回声从互联网上抓取的网络数据，这部分原始数据约100T，挑选出去重后的高质量中文数据，涉及到百科，书籍，博客，新闻，公告，小说，公众号等高质量长文本数据。这部分数据还在清洗更多逐步加入到模型中
- 中文Wikipedia的数据
- 中文悟道开源的200G数据
- clue开放的中文预训练数据，进行清洗后的高质量中文长文本数据
- 近年来中文自然语言处理多任务竞赛数据集，约150个
- [MNBVC](https://github.com/esbatmop/MNBVC) 中清洗出来的部分数据集
2. 其他语言数据（以英文为主）
- wiki_en
- openwebtext
- c4
3. 代码数据
为了能够提高模型的代码生成能力，我们添加了🤗开源的大量代码数据集
- codeparrot/github-code-clean
- codeparrot/apps
- huggingface-course/codeparrot-ds-train
- code_search_net
- Bigcode-the-stack-dedup
4. 持续更新
希望大家如果有较高质量的数据集能够提供给我们，不胜感激

## 模型下载
可以在🤗Model Hub下载以下所有模型

### 预训练模型

AtomGPT预训练模型使用transformers 直接加载就可以。4bit压缩模型需要使用[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ/blob/main/README_zh.md)进行加载

模型名称|🤗模型加载名称|下载地址
--|--|--
atomgpt-checkpoint-10k|atomgptai/atomgpt_checkpoint_10k|[模型下载](https://huggingface.co/AtomEchoAI/AtomGPT)

### chat模型
AtomGPT-chat模型需要使用transformers以及peft进行加载。4bit压缩版本模型需要使用[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ/blob/main/README_zh.md)进行加载

模型名称|🤗模型加载名称|下载地址
--|--|--
atomgpt-checkpoint-10k-chat|atomgptai/atomgpt_checkpoint_10k-chat|[模型下载](https://huggingface.co/AtomEchoAI/AtomGPT)

## 本地推理与快速部署

### 推理硬件要求

### gradio快速搭建问答平台

### docker部署问答接口

### transformers调用代码示例

## 常见问题列表

## 局限性

## 致谢

## 免责声明

## 问题反馈
