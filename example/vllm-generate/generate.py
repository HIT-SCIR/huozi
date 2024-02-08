from vllm import LLM, SamplingParams

prompts = [
    """<|beginofutterance|>系统
你是一个智能助手<|endofutterance|>
<|beginofutterance|>用户
请你用python写一段快速排序的代码<|endofutterance|>
<|beginofutterance|>助手
""",
]

sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, stop_token_ids=[57001], max_tokens=2048
)
llm = LLM(
    model="HIT-SCIR/huozi3",
    tensor_parallel_size=4,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(generated_text)
