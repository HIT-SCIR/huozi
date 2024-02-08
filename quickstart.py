import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HIT-SCIR/huozi3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

text = """<|beginofutterance|>系统
你是一个智能助手<|endofutterance|>
<|beginofutterance|>用户
请你用python写一段快速排序的代码<|endofutterance|>
<|beginofutterance|>助手
"""

inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(
    **inputs,
    eos_token_id=57001,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
