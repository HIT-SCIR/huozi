import re
from typing import List, Optional, Tuple

import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

DEBUG = False


class Huozi:
    def __init__(self, model_name_or_path: str, precision: str):
        torch_dtype = self.validate_device_precision(
            device="gpu" if torch.cuda.is_available() else "cpu",
            precision=precision,
        )

        self.model = HuoziForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            load_in_8bit=True if torch_dtype == torch.int8 else False,
        )
        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name_or_path, use_fast=True)

        self.im_end_id = self.tokenizer(self.model.im_end_token)["input_ids"][0]

    @staticmethod
    def validate_device_precision(device: str, precision: str):
        assert device in ["cpu", "gpu"]
        assert precision in ["fp32", "fp16", "bf16", "int8"]

        if device == "cpu":
            assert precision in ["fp32", "fp16"], "Currently Huozi only supports fp32 and fp16 on CPU."

        if precision != "fp16":
            print(f"Warning: Huozi checkpoint is saved in fp16, your choice ({precision}) may harm performance.")

        precision_torch_dtype = {
            "int8": torch.int8,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }

        return precision_torch_dtype[precision]

    def chat(self, generate_kwargs: dict, query: str, history: Optional[List[Tuple[str, str]]] = None):
        generate_kwargs["eos_token_id"] = self.im_end_id
        return self.model.chat(self.tokenizer, generate_kwargs, query, history=history)


class HuoziForCausalLM(BloomForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.im_start_token = "<|beginofutterance|>"
        self.im_end_token = "<|endofutterance|>"

        self.system_prompt = """你是由哈尔滨工业大学--自然语言处理研究所进行训练和部署的人工智能（Artificial Intelligence, AI）助手。
你的名字是“活字”。
你要为用户提供高质量的自然语言处理服务，旨在实现与用户之间的流畅、自然、可信、可靠和可用的对话。
你的目标是通过对话回答用户的问题、提供相关信息和建议，并能够执行各种任务，以满足用户的需求和期望。
你需要努力确保我们的服务能够提供准确、有用和全面的解决方案，以使用户获得最佳的体验和价值"""

        self.empty_response = "不好意思，我没有听明白。"

        self.system_name = "系统"
        self.user_name = "用户"
        self.bot_name = "助手"

        self.max_cycle_num = 15
        self.max_context_length = 1800
        self.max_total_tokens = 2048

    def process_response(self, response: str):
        # process_response is from chatglm-6b-int4, see:
        # https://huggingface.co/THUDM/chatglm-6b-int4/blob/6c5205c47d0d2f7ea2e44715d279e537cae0911f/modeling_chatglm.py#L1251

        response = response.strip()
        response = response.replace(self.im_start_token, "")
        response = response.replace(self.im_end_token, "")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    def dialog_to_chatml(self, dialogs: List[Tuple[str, Optional[str]]]):
        chatml_str = ""
        for query, response in dialogs:
            chatml_str += self.im_start_token + self.user_name + "\n" + query + self.im_end_token + "\n"
            chatml_str += self.im_start_token + self.bot_name + "\n" + response + self.im_end_token + "\n"
        return chatml_str

    def system_to_chatml(self):
        return self.im_start_token + self.system_name + "\n" + self.system_prompt + self.im_end_token + "\n"

    def query_to_chatml(self, query: str):
        chatml_str = ""
        chatml_str += self.im_start_token + self.user_name + "\n" + query + self.im_end_token + "\n"
        chatml_str += self.im_start_token + self.bot_name + "\n"
        return chatml_str

    @torch.inference_mode()
    def chat(self, tokenizer, generate_kwargs: dict, query: str, history: Optional[List[Tuple[str, str]]] = None):
        if history is None:
            history = []

        while True:
            prompt = self.system_to_chatml() + self.dialog_to_chatml(history) + self.query_to_chatml(query)
            inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
            if len(inputs["input_ids"][0]) <= self.max_context_length and len(history) < self.max_cycle_num:
                break
            history = history[1:]

        if DEBUG:
            print(f"==============prompt==============")
            print(prompt)
            print(f"==================================")

        generate_kwargs["max_new_tokens"] = min(self.max_total_tokens - len(inputs["input_ids"][0]),
                                                generate_kwargs["max_new_tokens"])
        outputs = self.generate(
            **inputs,
            **generate_kwargs,
        )
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)

        if len(response) == 0:
            response = self.empty_response

        if DEBUG:
            print(f"==============outputs==============")
            print(tokenizer.decode(outputs, skip_special_tokens=False))
            print(f"===================================")

        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history
