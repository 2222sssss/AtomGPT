## 🎇model_quantization介绍
用于模型量化，包括quantize_model.py和quantize.sh脚本文件

修改相关参数后在终端进入此目录下运行以下代码开始量化:
```
bash quantize.sh
```
### quantize_model.py
ScriptArguments数据类中是需要接收的参数,包括模型文件地址和数据文件地址及量化所需的各个参数

load_dataset方法用来处理所需的数据文件，传入数据文件路径和分词器，返回经过分词的数据样本

```
examples.append(tokenizer(text_item,max_length=2048,truncation=True))
```
上面代码默认最大文本长度为2048，可根据需要进行修改

```
model = AutoGPTQForCausalLM.from_pretrained(script_args.model_folder, quantize_config,max_memory={0:'24GIB',1:'24GIB',2:'24GIB','cpu':'80GIB'})
```
上面代码设置了最大的内存大小，可根据使用者实际情况进行修改

```
model.save_quantized(quantized_model_dir)
```
上面代码的参数是量化后模型保存的位置，通过脚本传入

### quantize.sh
bash脚本,您需要在此修改对应的参数

|参数|描述|
|----|----|
|model_folder|需要量化的模型位置，量化后的模型也会存放在此|
|example_file_path|所需数据的位置，可输入多条数据源|
|bits|量化后的比特数|

## 👻data_prepro介绍
准备训练的数据
### convert_for_atomhub
将下载的JSON文件转化为训练所需要的CSV文件

下面代码修改参数后在控制台执行:
```
python3 ./train/data_prepro/convert_for_atomhub.py 
--file_path ./financial_xiaozhu/xiaozhu.json 
--training_data_path ./financial_xiaozhu/training.csv 
--dev_data_path ./financial_xiaozhu/test.csv 
--num_dev 100
```
file_path为初始JSON文件，training_data_path和dev_data_path为转换后的CSV文件，前者为训练数据，后者为验证数据，num_dev是验证数据的数量



## 🎉merge_peft_model介绍
用于模型融合,可将微调得到的模型与基座模型融合
### merge_muilt_peft_adapter.py
多模型融合，可融合多个模型，在脚本文件中修改相关参数后进入merge_peft_model目录下，在终端运行以下代码执行：
```
bash merge_muilt.sh
```
### merge_muilt.sh
merge_muilt_peft_adapter.py的脚本文件，您需要在此修改相关参数
|参数|描述|
|----|----|
|adapter_model_name|需要融合的模型，可输入多个|
|output_name|融合后模型的名称|

### merge_peft_adapter.py
在脚本文件中修改相关参数后进入merge_peft_model目录下，在终端运行以下代码执行：
```
bash merge.sh
```

### merge.sh
merge_peft_adapter.py的脚本文件，您需要在此修改相关参数
|参数|描述|
|----|----|
|adapter_model_name|要融合的模型|
|output_name|融合后模型的名称|
|load8bit|是否加载8bit|
|tokenizer_fast|是否使用tokenizer_fast|

## ✨sft介绍
主要用于Lora微调
### finetune_cls_lora.py
lora微调代码，修改对应脚本后，进入到sft目录，在控制台执行下面代码：
```
bash finetune_other.sh
```
可将微调完成后在output_model目录里生成的模型用上面的融合代码与基座模型融合
### finetune_other.sh
finetune_cls_lora.py的脚本文件，您需要在此修改相关参数

最开头的output_model写上做完微调所生成的文件的存放位置
|参数|描述|
|----|----|
|localhost|此处可修改显卡号|
|model_name_or_path|要微调的模型|
|train_files|训练数据位置|
|validation_files|验证数据位置|
|do_train|开关，是否训练|
|do_eval|开关，是否做评估|
|use_fast_tokenizer|是否使用fast_tokenizer|
|output_dir|生成文件位置，开头写上|
|max_eval_samples|最大评估样本数|
|learning_rate|学习率|
|gradient_accumulation_steps|梯度累积次数|
|num_train_epochs|训练轮数|
|warmup_steps|学习率预热步数|
|lora_r|[Lora](https://zhuanlan.zhihu.com/p/620552131)的r参数|
|lora_alpha|Lora的alpha参数|
|logging_dir|生成的日志文件地址，默认在output_model的logs目录下|

