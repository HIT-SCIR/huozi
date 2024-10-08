<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  <img src="image/huozi-logo.jpg" width="30" /> 活字通用大模型
</h1>
</div>
</p>

<div align="center">
<a href="https://github.com/HIT-SCIR/huozi/pulls">
<image src="https://img.shields.io/badge/PRs-welcome-brightgreen">
</a>
<a href="https://github.com/HIT-SCIR/huozi/pulls">
<image src="https://img.shields.io/badge/License-Apache_2.0-green.svg">
</a>
<!-- <h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/HIT-SCIR/huozi/blob/main/README_EN.md">English</a>
    <p>
</h4> -->
</div>

## 🎉 更新

- [2024-09-12] 发布活字3.5版本
- [2024-02-09] 发布活字3.5版本和中文MT-Bench数据集
- [2023-08-06] 发布活字1.0和活字2.0版本
- [2023-05-04] 发布《ChatGPT调研报告》；内测活字大模型

## 🔖 目录

|章节|说明|
|---|---|
|[💁🏻‍♂ 开源清单](#-开源清单)|本仓库开源项目清单|
|[💡 模型介绍](#-模型介绍)|简要介绍活字模型结构和训练过程|
|[📥 模型下载](#-模型下载)|活字模型下载链接|
|[💻 模型推理](#-模型推理)|活字模型推理样例，包括vLLM、llama.cpp、Ollama等推理框架的使用流程|
|[📈 模型性能](#-模型性能)|活字模型在主流评测任务上的性能|
|[🗂 生成样例](#-生成样例)|活字模型实际生成效果样例|

## 💁🏻‍♂ 开源清单
![](image/models-v3.5.png)
<!-- - **活字 3.5**: [[模型权重](#-模型下载)] [[在线Demo](https://huozi.8wss.com)] -->
- **活字 3.5**: [[模型权重](#-模型下载)]
    - 活字3.5为基于活字3.0和Chinese-Mixtral-8x7B进行进一步性能优化的新模型。
- **活字 3.0**: [[模型权重](#-模型下载)] [[在线Demo](https://huozi.8wss.com)]
    - 活字3.0为一个稀疏混合专家模型，支持32K上下文，具有丰富的中、英文知识和强大的数学推理、代码生成能力。活字3.0较旧版活字具有更强的指令遵循能力和安全性。
- **中文MT-Bench**: [[数据集](data/mt-bench-zh/)]
    - 本数据集是英文MT-Bench对话能力评测数据集的中文版。它包含了一系列多轮对话问题，每一组问题都经过了精心的人工校对，并为适应中文语境进行了必要的调整。
- **《ChatGPT 调研报告》**: [[PDF](https://github.com/HIT-SCIR/huozi/blob/main/pdf/chatgpt_book.pdf)]
    - 哈工大自然语言处理研究所组织多位老师和同学撰写了本调研报告，从技术原理、应用场景、未来发展等方面对ChatGPT进行了尽量详尽的介绍及总结。
- **活字 2.0**: [[模型权重](https://huggingface.co/HIT-SCIR/huozi-7b-rlhf)] [[RLHF数据](data/huozi-rlhf/huozi_rlhf_data.csv)]
    - 在活字1.0基础上，通过人类反馈的强化学习（RLHF）进一步优化了模型回复质量，使其更加符合人类偏好。相较于上一个版本平均长度明显提高，遵从指令的能力更强，逻辑更加清晰。
    - 16.9k 人工标注的偏好数据，回复来自活字模型，可以用于训练奖励模型。
- **活字 1.0**: [[模型权重](https://huggingface.co/HIT-SCIR/huozi-7b-sft)]
    - 在Bloom模型的基础上，在大约 150 亿 tokens 上进行指令微调训练得到的模型，具有更强的指令遵循能力、更好的安全性。

## 💡 模型介绍

大规模语言模型（LLM）在自然语言处理领域取得了显著的进展，并在广泛的应用场景中展现了其强大的潜力。这一技术不仅吸引了学术界的广泛关注，也成为了工业界的热点。在此背景下，哈尔滨工业大学社会计算与信息检索研究中心（HIT-SCIR）近期推出了最新成果——**活字3.5**，致力于为自然语言处理的研究和实际应用提供更多可能性和选择。

活字3.5是在[活字3.0](https://github.com/HIT-SCIR/huozi/README-v3.md)和[Chinese-Mixtral-8x7B](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)基础上，进行进一步性能增强得到的模型。活字3.5支持**32K长上下文**，继承了活字3.0强大的综合能力，并在**中英文知识**、**数学推理**、**代码生成**、**指令遵循能力**、**内容安全性**等诸多方面实现了性能提升。

> [!IMPORTANT]
> 活字系列模型仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用生成的内容，请勿将生成的有害内容传播至互联网。

> 活字1.0和活字2.0的文档请见[此处](README-v1v2.md)。
> 活字3.0和中文MT-Bench的文档请见[此处](README-v3.md)。

### 模型结构

活字3.5是一个稀疏混合专家模型（SMoE），每个专家层包含8个FFN，每次前向计算采用top-2稀疏激活。活字3.5共拥有46.7B参数，得益于其稀疏激活的特性，实际推理时仅需激活13B参数，有效提升了计算效率和处理速度。

<!-- ![](image/smoe-v3.5.png) -->
<p align = "center">
    <img src="image/smoe-v3.5.png" width="300" />
</p>

### 训练过程

活字3.5经过了多步训练，如下图所示：

![](image/train-process-v3.5.png)

其训练过程为：
1. 【中文扩词表增量预训练】： 由于Mixtral-8x7B词表不支持中文，因此对中文的编解码效率较低，限制了中文场景下的实用性。我们首先基于Mixtral-8x7B进行了中文扩词表增量预训练，显著提高了模型对中文的编解码效率，并使模型具备了强大的中文生成和理解能力。我们已于[Chinese-Mixtral-8x7B代码仓库](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)开源了模型权重和训练代码。
2. 【活字3.0训练】：我们基于Chinese-Mixtral-8x7B在大约30万行指令数据上进行微调，得到了活字3.0模型，使用的数据集见[此处说明](https://github.com/HIT-SCIR/huozi/issues/11#issuecomment-1973113428)。活字3.0继承了基座模型丰富的中英文知识，并在数学推理、代码生成等任务上具有强大性能。经过指令微调，活字3.0还在指令遵循能力和安全性方面实现了显著提升。
3. 【活字1.0数据集微调】：我们尝试使用活字1.0数据集对Chinese-Mixtral-8x7B进行指令微调，得到的*中间检查点 1*在中英文知识（如 C-Eval、CMMLU、MMLU 等任务）方面表现优异，甚至超过了活字3.0。然而，该模型在指令遵循能力和安全性上落后活字3.0较多。
4. 【指令遵循能力强化】：针对*中间检查点 1*在指令遵循能力上的不足，我们引入了额外的数据集进行强化。此外，根据[Longxu Dou等人的经验](https://arxiv.org/pdf/2404.03608)，我们在训练过程中使用了[BPE Dropout](https://aclanthology.org/2020.acl-main.170/)技术，以进一步增加模型对指令的鲁棒性。该过程训练得到了*中间检查点 2*。
5. 【模型融合】：我们参考[Yiming Cui等人的方法](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)，对*中间检查点 1*、*中间检查点 2*以及活字3.0模型进行融合，生成了*中间检查点 3*。
6. 【模型融合后训练】：在融合后的模型基础上，我们进一步进行了指令微调，最终推出了活字3.5。该版本在中英文知识、指令遵循能力和安全性回复等方面均有提升。

## 📥 模型下载

|模型名称|文件大小|下载地址|备注|
|:---:|:---:|:---:|:---:|
|huozi3.5|88GB|[🤗HuggingFace](https://huggingface.co/HIT-SCIR/huozi3.5)<br>[ModelScope](https://modelscope.cn/models/HIT-SCIR/huozi3.5/summary)|活字3.5 完整模型|
|huozi3.5-ckpt-1|88GB|[🤗HuggingFace](https://huggingface.co/HIT-SCIR/huozi3.5-ckpt-1)<br>[ModelScope](https://modelscope.cn/models/HIT-SCIR/huozi3.5-ckpt-1/summary)|活字3.5 中间检查点 1|
|huozi3.5-ckpt-2|88GB|[🤗HuggingFace](https://huggingface.co/HIT-SCIR/huozi3.5-ckpt-2)<br>[ModelScope](https://modelscope.cn/models/HIT-SCIR/huozi3.5-ckpt-2/summary)|活字3.5 中间检查点 2|
|huozi3.5-ckpt-3|88GB|[🤗HuggingFace](https://huggingface.co/HIT-SCIR/huozi3.5-ckpt-3)<br>[ModelScope](https://modelscope.cn/models/HIT-SCIR/huozi3.5-ckpt-3/summary)|活字3.5 中间检查点 3|

如果您希望微调活字3.5或Chinese-Mixtral-8x7B，请参考[此处训练代码](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B?tab=readme-ov-file#%E5%BE%AE%E8%B0%83)。

## 💻 模型推理

### Quick Start

活字3.5采用ChatML格式的prompt模板，格式为：
```
<|beginofutterance|>系统
{system prompt}<|endofutterance|>
<|beginofutterance|>用户
{input}<|endofutterance|>
<|beginofutterance|>助手
{output}<|endofutterance|>
```

使用活字3.5进行推理的示例代码如下：
```python
# quickstart.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HIT-SCIR/huozi3.5"

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
```

活字3.5支持全部Mixtral模型生态，包括Transformers、vLLM、llama.cpp、Ollama、Text generation web UI等框架。

如果您在下载模型时遇到网络问题，可以使用我们在[ModelScope](#modelscope-模型推理)上提供的检查点。

<details>
<summary>

#### Transformers 模型推理 + 流式生成

</summary>

transformers支持为tokenizer添加聊天模板，并支持流式生成。示例代码如下：
```python
# example/transformers-stream/stream.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "HIT-SCIR/huozi3.5"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
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
```

</details>

<details>
<summary>

#### ModelScope 模型推理

</summary>

ModelScope的接口与Transformers非常相似，只需将transformers替换为modelscope即可：
```diff
# example/modelscope-generate/generate.py

import torch
- from transformers import AutoModelForCausalLM, AutoTokenizer
+ from modelscope import AutoTokenizer, AutoModelForCausalLM

model_id = "HIT-SCIR/huozi3.5"

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
```

</details>

<details>
<summary>

#### vLLM 推理加速

</summary>

活字3.5支持通过vLLM实现推理加速，示例代码如下：
```python
# example/vllm-generate/generate.py

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
    model="HIT-SCIR/huozi3.5",
    tensor_parallel_size=4,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(generated_text)
```

</details>

<details>
<summary>

#### 部署 OpenAI API Server

</summary>

活字3.5可以部署为支持OpenAI API协议的服务，这使得活字3.5可以直接通过OpenAI API进行调用。

环境准备：
```shell
$ pip install vllm openai
```

启动服务：
```shell
$ python -m vllm.entrypoints.openai.api_server --model /path/to/huozi3.5/checkpoint --served-model-name huozi --chat-template template.jinja --tensor-parallel-size 8 --response-role 助手 --max-model-len 2048
```

使用OpenAI API发送请求：
```python
# example/openai-api/openai-client.py

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="huozi",
    messages=[
        {"role": "系统", "content": "你是一个智能助手"},
        {"role": "用户", "content": "请你用python写一段快速排序的代码"},
    ],
    extra_body={"stop_token_ids": [57001]},
)
print("Chat response:", chat_response.choices[0].message.content)
```

下面是一个使用OpenAI API + Gradio + 流式生成的示例代码：
```python
# example/openai-api/openai-client-gradio.py

from openai import OpenAI
import gradio as gr

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(message, history):
    history_openai_format = [
        {"role": "系统", "content": "你是一个智能助手"},
    ]
    for human, assistant in history:
        history_openai_format.append({"role": "用户", "content": human})
        history_openai_format.append({"role": "助手", "content": assistant})
    history_openai_format.append({"role": "用户", "content": message})
    models = client.models.list()

    stream = client.chat.completions.create(
        model=models.data[0].id,
        messages=history_openai_format,
        temperature=0.8,
        stream=True,
        extra_body={"repetition_penalty": 1, "stop_token_ids": [57001]},
    )

    partial_message = ""
    for chunk in stream:
        partial_message += chunk.choices[0].delta.content or ""
        yield partial_message


gr.ChatInterface(predict).queue().launch()
```

</details>

### 量化推理

<details>
<summary>

#### GGUF 格式

</summary>

GGUF格式旨在快速加载和保存模型，由llama.cpp团队推出，适用于llama.cpp、Ollama等框架。您可以手动将HuggingFace格式的活字3.5转换到GGUF格式。

##### Step 1 环境准备

首先需要下载llama.cpp的源码。我们在仓库中提供了llama.cpp的submodule，这个版本的llama.cpp已经过测试，可以成功进行推理：
```shell
$ git clone --recurse-submodules https://github.com/HIT-SCIR/huozi
$ cd examples/llama.cpp
```

您也可以下载最新版本的llama.cpp源码：
```shell
$ git clone https://github.com/ggerganov/llama.cpp.git
$ cd llama.cpp
```

然后需要进行编译。根据您的硬件平台，编译命令有细微差异：
```shell
$ make  # 用于纯CPU推理
$ make LLAMA_CUBLAS=1  # 用于GPU推理
$ LLAMA_METAL=1 make  # 用于Apple Silicon，暂未经过测试
```

##### Step 2 格式转换（可选）

以下命令需要在`llama.cpp/`目录下：
```shell
# 转换为GGUF格式
$ python convert.py --outfile /path/to/huozi-gguf/huozi3.5.gguf /path/to/huozi3.5
# 进行GGUF格式的q4_0量化
$ quantize /path/to/huozi-gguf/huozi3.5.gguf /path/to/huozi-gguf/huozi3.5-q4_0.gguf q4_0
```

##### Step 3 开始推理

以下命令需要在`llama.cpp/`目录下：
```shell
$ main -m /path/to/huozi-gguf/huozi3.5-q4_0.gguf --color --interactive-first -c 2048 -t 6 --temp 0.2 --repeat_penalty 1.1 -ngl 999 --in-prefix "<|beginofutterance|>用户\n" --in-suffix "<|endofutterance|>\n<|beginofutterance|>助手" -r "<|endofutterance|>"
```

`-ngl`参数表示向GPU中offload的层数，降低这个值可以缓解GPU显存压力。经过我们的实际测试，q2_k量化的模型offload 16层，显存占用可降低至9.6GB，可在消费级GPU上运行模型：
```shell
$ main -m /path/to/huozi-gguf/huozi3.5-q2_k.gguf --color --interactive-first -c 2048 -t 6 --temp 0.2 --repeat_penalty 1.1 -ngl 16 --in-prefix "<|beginofutterance|>用户\n" --in-suffix "<|endofutterance|>\n<|beginofutterance|>助手" -r "<|endofutterance|>"
```

关于`main`的更多参数，可以参考llama.cpp的[官方文档](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)。

使用Ollama框架进行推理，可以参考Ollama的[README说明](https://github.com/ollama/ollama#import-from-gguf)。

</details>

## 📈 模型性能

![](image/metric-v3.5-h.png)

针对大模型综合能力评价，我们分别使用以下评测数据集对活字3.5进行评测：
- C-Eval：一个全面的中文基础模型评估套件。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。
- CMMLU：一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力，涵盖了从基础学科到高级专业水平的67个主题。
- GAOKAO：一个以中国高考题目为数据集，旨在提供和人类对齐的，直观，高效地测评大模型语言理解能力、逻辑推理能力的测评框架。
- MMLU：一个包含57个多选任务的英文评测数据集，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平，是目前主流的LLM评测数据集之一。
- HellaSwag：一个极具挑战的英文NLI评测数据集，每一个问题都需要对上下文进行深入理解，而不能基于常识进行回答。
- GSM8K：一个高质量的小学数学应用题的数据集，这些问题需要 2 到 8 个步骤来解决，解决方案主要涉及使用基本算术运算，可用于评价多步数学推理能力。
- HumanEval：一个由 164 个原创编程问题组成的数据集，通过衡量从文档字符串生成程序的功能正确性，来够评估语言理解、算法和简单的数学能力。
- MT-Bench：一个开放的英文问题集，包括80个多轮对话任务，用于评估聊天机器人的多轮对话和指令遵循能力，并通过大模型裁判（GPT-4）对模型回答进行打分。
- MT-Bench-zh：我们根据MT-Bench翻译得来的中文问题集，每组问题均经过人工校对和中文语境下的适当调整。我们已在[此处](data/mt-bench-zh/)开源MT-Bench-zh数据集。
- MT-Bench-safety：我们手工构造的安全数据集，包括暴力、色情、敏感等风险内容。该数据集为封闭数据集。

活字3.5在推理时仅激活13B参数。下表为活字3.5与其他13B规模的中文模型以及旧版活字在各个评测数据集上的结果：

![](image/evaluation-v3.5.png)

> 我们在C-Eval、CMMLU、MMLU采用5-shot，GSM8K采用4-shot，HellaSwag、HumanEval采用0-shot，HumanEval采用pass@1指标。所有测试均采用greedy策略。
>
> 我们使用OpenCompass作为评测框架，commit hash为[4c87e77](https://github.com/open-compass/opencompass/tree/4c87e777d855636b9eda7ec87bcbbf12b62caed3)。评测代码位于[此处](./evaluate/)。
>
> 在活字3.0的性能评测中，我们在HumanEval错误使用了base模型的评测方法，正确的评测结果已在上表内更新。

根据上表中的测试结果，活字3.5较活字3.0取得了较稳定的性能提升，活字3.5的**中英文知识**、**数学推理**、**代码生成**、**中文指令遵循能力**、**中文内容安全性**等多方面能力均得到了加强。

## 🗂 生成样例

下面是活字3.5在MT-Bench-zh评测集上的生成效果展示：

![](image/examples/v3.5-case1.png)
![](image/examples/v3.5-case2.png)
![](image/examples/v3.5-case3.png)
![](image/examples/v3.5-case4.png)
![](image/examples/v3.5-case5.png)
![](image/examples/v3.5-case6.png)

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/folders.png" width="25" /> 开源协议
对本仓库源码的使用遵循开源许可协议 [Apache 2.0](https://github.com/HIT-SCIR/huozi/blob/main/LICENSE)。

活字支持商用。如果将活字模型或其衍生品用作商业用途，请您按照如下方式联系许可方，以进行登记并向许可方申请书面授权：联系邮箱：<jngao@ir.hit.edu.cn>。

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/notes.png" width="25" /> Citation

### 活字大模型

```latex
@misc{huozi,
    author = {Huozi-Team}.
    title = {Huozi: Leveraging Large Language Models for Enhanced Open-Domain Chatting}
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository}
    howpublished = {\url{https://github.com/HIT-SCIR/huozi}}
}
```

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/star.png" width="25" /> Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HIT-SCIR/huozi&type=Date)](https://star-history.com/#HIT-SCIR/huozi&Date)
