## GitHub仓库克隆到本地
[Git克隆](https://blog.csdn.net/wlq_567/article/details/127424297)

[VScode克隆](https://blog.csdn.net/rrmod/article/details/128975123)

## conda环境安装
在运行前需要安装conda环境，以及外面的requirements.txt中的资源包也要安装好。

[Linux安装conda环境](https://blog.csdn.net/qq_44173974/article/details/125336916)

[Windows安装conda环境](https://blog.csdn.net/weixin_42421914/article/details/130092613)

requirements.txt资源包安装示例,在自己创建的环境终端运行
```
pip install bitsandbytes==0.39.0
```

## conda常用命令
```
# 创建虚拟环境(name是自己起的环境名字)
conda create -n name python==3.7
# 激活环境
conda activate name
# 退出环境
conda deactivate
# 查看虚拟环境
conda info --envs
# 删除虚拟环境
conda remove -n name --all
# 删除所有的安装包及cache(索引缓存、锁定文件、未使用过的包和tar包)
conda clean -y --all
 
```


## atomgpt_chat.py介绍
用于gradio搭建问答界面，可进行对话，在控制台输入以下代码加载界面，4bit的则输入第二条代码，注意最后面的模型名字，若是换一个模型只需要换个名字就行了，huggingface左上角可以直接复制模型名字。
运行之后在网页上可以与atomgpt对话。

```
python examples/atomgpt_chat.py --model_name_or_path AtomEchoAI/AtomGPT_28k
```

```
python examples/atomgpt_chat.py --model_name_or_path AtomEchoAI/AtomGPT_28k_chat_4bit --is_4bit
```

## atomgpt_for_langchain.py介绍
封装了AtomGPT的加载和输入输出，用于给第三个python文件atomgpt_with_search导入，由于langchain版本问题，若是报错:
```
TypeError: Can't instantiate abstract class AtomGPT with abstract methods __call__
```
可尝试将代码里的_call更改为__call__

注意更改之后要先运行一下，不然被导入的还是未更新的


## atomgpt_with_search.py介绍

在下面语句中输入你想运行的模型版本名称
```
llm = AtomGPT('AtomEchoAI/AtomGPT_14k_chat_4bit')
```

在下面语句中输入你的提问

```
response = answer_from_web('北京在哪？',llm)
```
此代码模块的功能是先在网上搜索你所提问的相关信息，将文本保留在context中，然后调用atomgpt_for_langchain文件中的_call方法，AtomGPT会根据网上搜索到的context给出你提问的答案。

代码是从外网搜索的context
```
from duckduckgo_search import ddg
```
若是报错模组缺失，请执行：
```
pip install duckduckgo_search
```

## gunicorn_atomgpt_server.py介绍

api接口

在控制台输入以下命令运行：
```
uvicorn examples.gunicorn_atomgpt_server:app --reload
```

可在控制台输入下面格式的命令：

```
curl http://127.0.0.1:8000/atomgpt_chat/ -X POST -H "Content-Type: application/json" --data '[{"content": [{'role':'Human','content':'介绍一下你自己'},{'role':'Assistant','content':'我是AtomGPT'}],"temperature":0.5,"top_p":0.5}]'
```

格式遵循下面的代码：
```
# message ： [{'role':'Human','content':'介绍一下你自己'},{'role':'Assistant','content':'我是AtomGPT'}]
class InputData(BaseModel):
    messages: list
    temperature: float
    top_p:float

```
