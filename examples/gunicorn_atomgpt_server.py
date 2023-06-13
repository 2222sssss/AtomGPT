from asyncio.log import logger
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel
from json import dumps
import time
app = FastAPI()
# message ： [{'role':'Human','content':'介绍一下你自己'},{'role':'Assistant','content':'我是AtomGPT'}]
class InputData(BaseModel):
    messages: list[dict]
    temperature: float
    top_p:float

@app.on_event("startup")
def setup():
    global model
    global tokenizer
    model_name_or_path = ''
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=False)
    is_4bit = True
    if is_4bit==False:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
        model.eval()
    else:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,low_cpu_mem_usage=True, device="cuda:0")
    logger.info('startup')    

@app.get("/")
def read_root():
    return {"AtomGPT : Hello \n api:atomgpt_chat"}


# separate from recommend
@app.post("/atomgpt_chat/")
def atomgpt_chat(input_data: InputData):
    global model
    global tokenizer
    prompt = ''
    for input_text_one in input_data.messages:
        prompt += "<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>"
    if input_data.messages[-1]['role']=='Human':
        prompt += "<s>Assistant: "
    else:
        prompt += "<s>Human: "
    prompt = prompt[-2048:]
    input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":1024,
        "do_sample":True,
        "top_k":50,
        "top_p":input_data.top_p,
        "temperature":input_data.temperature,
        "repetition_penalty":1.2,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
    }
    generate_ids = model.generate(**generate_input)
    generate_ids = [item[len(input_ids[0]):-1] for  item in generate_ids]
    result_message = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
    return result_message
