#  CUDA_VISIBLE_DEVICES=0 python Medgemma.py --methed volucompose

import argparse
import re
import torch
from transformers import AutoProcessor, AutoConfig
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import os,json

MODEL_PATH="../workspace/model/medgemma-1.5-4b"

TEMPERATURE=0
TOP_P=0.95
MAX_NEW_TOKENS=300
REPETITION_PENALTY=1.0
MAX_IMAGE_NUM=600

parser = argparse.ArgumentParser()
parser.add_argument(
  "--methed",
  default=os.environ.get("METHED", "orignal"),
  choices=["orignal", "volucompose"],
  help="Compression methed. Can also be provided by env var METHED.",
)
args = parser.parse_args()

def _load_model_kwargs(methed):
    config_path = "../config/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if methed == "volucompose":
        print(f"run {methed}")
        from medcompose.volucompose import Gemma3ForConditionalGeneration
        return config_data.get(methed, {}),Gemma3ForConditionalGeneration
    elif methed == "holitom":
        print(f"run {methed}")
        from medcompose.holitom import Gemma3ForConditionalGeneration
        return config_data.get(methed, {}),Gemma3ForConditionalGeneration
    elif methed == "unicomp":
        print(f"run {methed}")
        from medcompose.unicomp import Gemma3ForConditionalGeneration
        return config_data.get(methed, {}),Gemma3ForConditionalGeneration
    elif methed == "visionzip":
        print(f"run {methed}")
        from medcompose.visionzip import Gemma3ForConditionalGeneration
        return config_data.get(methed, {}),Gemma3ForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        return {},AutoModelForImageTextToText

video_dir = "../data/Eval/Amos/amos_test_crop_32/amos_0034/"
prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."

image_files = sorted([
  f for f in os.listdir(video_dir) 
  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.dcm'))
])

model_kwargs,model_class = _load_model_kwargs(args.methed)
model = model_class.from_pretrained(
  MODEL_PATH,
  trust_remote_code=True,
  torch_dtype=torch.bfloat16,
  device_map="cuda",
  attn_implementation="flash_attention_2",
  **model_kwargs
)
  
processor = AutoProcessor.from_pretrained(
  MODEL_PATH,
  trust_remote_code=True
)
tokenizer = processor.tokenizer
model.eval()

temperature = TEMPERATURE
top_p = TOP_P
max_new_tokens = MAX_NEW_TOKENS
repetition_penalty = REPETITION_PENALTY

current_messages = []
content = []
for i,image in enumerate(image_files):
  if isinstance(image,str):
    image = Image.open(f"{video_dir}/{image}")
  content.append({"type":"text","text":f"SLICE {i+1}: "})
  content.append({"type":"image","image":image})
content.append({"type":"text","text":prompt})
current_messages.append({"role":"user","content":content})

inputs = processor.apply_chat_template(
  current_messages, add_generation_prompt=True, tokenize=True,
  return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]
print(f"{'='*10}start generate{'='*10}")
with torch.inference_mode():
  do_sample = True if temperature > 0 else False
  generation = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    pad_token_id=processor.tokenizer.eos_token_id,
    top_k = None,top_p = None
  )
  generation = generation[0][input_len:]

outputs = processor.decode(generation, skip_special_tokens=True)
print(f"result:{outputs}")
