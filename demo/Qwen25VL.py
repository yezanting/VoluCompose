#  CUDA_VISIBLE_DEVICES=0 python Qwen25VL.py --methed medpruner++

import argparse
import torch
from transformers import AutoProcessor, AutoConfig

import os,json
from qwen_vl_utils import process_vision_info

MODEL_PATH="../workspace/model/Qwen2.5-VL-7B-Instruct"

TEMPERATURE=0
TOP_P=0.95
MAX_NEW_TOKENS=600
REPETITION_PENALTY=1.0
MAX_IMAGE_NUM=600

parser = argparse.ArgumentParser()
parser.add_argument(
  "--methed",
  default=os.environ.get("METHED", "orignal"),
  choices=["orignal", "medpruner++"],
  help="Compression methed. Can also be provided by env var METHED.",
)
args = parser.parse_args()

def _load_model_kwargs(methed):
    config_path = "../config/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if methed == "medpruner++":
        print(f"run {methed}")
        from medcompose.medpruner_pp import Qwen2_5_VLForConditionalGeneration
        return config_data.get(methed, {}),Qwen2_5_VLForConditionalGeneration
    elif methed == "holitom":
        print(f"run {methed}")
        from medcompose.holitom import Qwen2_5_VLForConditionalGeneration
        return config_data.get(methed, {}),Qwen2_5_VLForConditionalGeneration
    elif methed == "unicomp":
        print(f"run {methed}")
        from medcompose.unicomp import Qwen2_5_VLForConditionalGeneration
        return config_data.get(methed, {}),Qwen2_5_VLForConditionalGeneration
    elif methed == "visionzip":
        print(f"run {methed}")
        from medcompose.visionzip import Qwen2_5_VLForConditionalGeneration
        return config_data.get(methed, {}),Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return {},Qwen2_5_VLForConditionalGeneration

video_dir = "../data/Eval/3DRad/valid_crop/valid_1000_a_1/"
prompt = "<video>\nDoes this CT image present medical material?\nA.Yes\nB.No"

image_files = sorted([
  video_dir + f for f in os.listdir(video_dir)
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

messages = []
content = []
for i,image in enumerate(image_files):
  content.append({"type":"image","image":image})
content.append({"type":"text","text":prompt})
messages.append({"role":"user","content":content})


prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

do_sample = False if temperature == 0 else True
generated_ids = model.generate(**inputs,temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,max_new_tokens=max_new_tokens,do_sample = do_sample)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
outputs = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"result:{outputs}")

