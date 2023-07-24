from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional,List
from transformers import AutoTokenizer, HfArgumentParser
import pandas as pd
import logging
import os


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_folder: Optional[str] = field(default=None, metadata={"help": "the model name"})
    example_file_path: Optional[List[str]] = field(default=None, metadata={"help": "the model name"})
    group_size : Optional[int] = field(default=128, metadata={"help": "the model type"})
    bits : Optional[int] = field(default=4, metadata={"help": "the model type"})
    tokenizer_fast:Optional[bool] = field(default=False, metadata={"help": "the model type"})
    quant_batch_size:Optional[int] = field(default=1, metadata={"help": "the model type"})
    use_triton:Optional[bool] = field(default=False, metadata={"help": "the model type"})
    samples_num : Optional[int] = field(default=3000, metadata={"help": "the samples_num"})
    
def load_dataset(file_path,tokenizer):
    data_all = []
    for file_path_one in file_path:
        data = pd.read_csv(file_path_one)
        data_all.append(data)
    data = pd.concat(data_all)
    print('输入数据行数,',data.shape[0])
    data = data.sample(n=script_args.samples_num)
    examples = []
    for text_item in tqdm(data['text'].to_list()):
        examples.append(tokenizer(text_item,max_length=2048,truncation=True))
    return examples
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained(script_args.model_folder, use_fast=script_args.tokenizer_fast)
examples = load_dataset(script_args.example_file_path,tokenizer)
quantize_config = BaseQuantizeConfig(
    bits=script_args.bits,  # 将模型量化为 4-bit 数值类型
    group_size=script_args.group_size,  # 一般推荐将此参数的值设置为 128
    desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
)
# max_memory={0:'24GIB','cpu':'80GIB'}
model = AutoGPTQForCausalLM.from_pretrained(script_args.model_folder, quantize_config,max_memory={0:'24GIB',1:'24GIB',2:'24GIB','cpu':'80GIB'})
print('load model over')
model.quantize(examples,batch_size=script_args.quant_batch_size,use_triton=script_args.use_triton,autotune_warmup_after_quantized=False,cache_examples_on_gpu=False)
quantized_model_dir = os.path.join(script_args.model_folder,'pytorch_model_'+str(script_args.bits)+'bit.model')
model.save_quantized(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)