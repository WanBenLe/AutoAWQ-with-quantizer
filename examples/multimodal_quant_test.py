import subprocess
import os
import logging
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,AutoProcessor
import torch
from PIL import Image
import requests
import pickle
import numpy as np
from awq.utils.utils import clear_memory, get_best_device


model_path = "llava-hf/llava-v1.6-34b-hf"
quant_path = "./llava-v1.6-34b-hf-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, torch_dtype=torch.float16, device_map="auto", use_flash_attention_2=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(model)
# Define data loading methods
def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

calib_data=pickle.load(open('inputs.pkl','rb'))
totensor=['attention_mask','position_ids','inputs_embeds']
best_device = get_best_device()

sample_size=32
for i in totensor:
    calib_data[i]= torch.Tensor(np.repeat(cdata[i],sample_size,axis=0))
print(calib_data)

# Quantize
quant_calib_data_type='multimodal'
if quant_calib_data_type=='multimodal':
    # multimodal calib_data
    model.quantize(tokenizer, quant_config=quant_config, samples=sample_size, calib_data=calib_data,
    calib_data_type='multimodal')
else:
    # text calib_data with samples and blocksize
    model.quantize(tokenizer, quant_config=quant_config,samples=768,blocksize=1024, calib_data=load_wikitext())


# Quantize
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
