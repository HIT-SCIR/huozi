from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer


model_id = "HIT-SCIR/huozi3-awq"

model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.chat_template = """{% for message in messages %}{{'<|beginofutterance|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|endofutterance|>' + '\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != '助手' %}{{ '<|beginofutterance|>助手\n' }}{% endif %}"""

chat = [
    {"role": "系统", "content": "你是一个智能助手"},
    {"role": "用户", "content": "请你用python写一段快速排序的代码"},
]

inputs = tokenizer.apply_chat_template(
    chat,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(0)

stream_output = model.generate(
    inputs,
    streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
    eos_token_id=57001,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
)
