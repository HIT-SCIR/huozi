import fire
import gradio as gr

from utils import Huozi


def run_cli(generate_kwargs, model):
    history = None
    while True:
        query = input(">>>> ")
        if query == "q":
            break

        response, history = model.chat(generate_kwargs, query, history=history)
        print(f"Bot: {response}")


def run_gradio(generate_kwargs, model):
    max_new_token_slider = gr.Slider(
        minimum=1,
        maximum=1024,
        value=generate_kwargs['max_new_tokens'],
        step=1,
        label="max_new_tokens",
        info="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
    )
    temperature_slider = gr.Slider(
        minimum=1e-3,
        maximum=5.000,
        value=generate_kwargs['temperature'],
        step=1e-3,
        label="temperature",
        info="The value used to modulate the next token probabilities."
    )
    repetition_penalty_slider = gr.Slider(
        minimum=1.00,
        maximum=5.00,
        value=generate_kwargs['repetition_penalty'],
        step=0.01,
        label="repetition_penalty",
        info="The value used to modulate the next token probabilities."
    )
    topk_slider = gr.Slider(
        minimum=1,
        maximum=500,
        value=generate_kwargs['top_k'],
        step=1,
        label="topk",
        info="The number of highest probability vocabulary tokens to keep for top-k-filtering."
    )
    topp_slider = gr.Slider(
        minimum=0.01,
        maximum=1.00,
        value=generate_kwargs['top_p'],
        step=0.01,
        label="topp",
        info=" If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
    )
    do_sample_checkbox = gr.Checkbox(
        value=generate_kwargs['do_sample'],
        label="do_sample",
        info=" Whether or not to use sampling ; use greedy decoding otherwise.",
    )

    def Config_Chat(query, history, max_new_tokens, temperature, repetition_penalty, topk, topp, do_sample):
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "top_k": topk,
            "top_p": topp,
        }
        # print(generate_kwargs)
        # print(query)
        # print(history)
        return model.chat(generate_kwargs, query, history)[0]

    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=Config_Chat,
            title="您好，我是活字",
            description="作为一个通用的人工智能语言模型，我能回答您的问题，帮您高效完成工作",
            # 每一个都需要放在一个list内
            examples=[
                ["去哈尔滨要准备什么东西？"],
                ["帮我写一个计算n的阶乘的代码"],
                ["我5月1号到5月7号要出国游玩，请帮我写一封五一的请假信"],
                ["如果运行程序出现segmentation fault，可能的原因是什么？如何避免？"]
            ],
            submit_btn="提交",
            retry_btn="重新生成",
            undo_btn="撤销",
            clear_btn="清空",
            additional_inputs=[max_new_token_slider, temperature_slider, repetition_penalty_slider, topk_slider,
                               topp_slider, do_sample_checkbox],
            additional_inputs_accordion_name="Generation Config"
        )

    # 本地运行 (如果要使用gradio生成分享链接，share=True)
    demo.launch(share=False)
    # 如果在服务器运行，如下声明后即可通过服务器地址+端口号的方式在其他设备访问
    demo.launch(server_name="0.0.0.0", server_port=7860， share=False)


def main(
        model_name_or_path: str = "HIT-SCIR/huozi-7b-sft",
        precision: str = "fp16",
        mode: str = "gradio",

        # default_generate_kwargs
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        do_sample: bool = True,
        repetition_penalty: float = 1.03,
        top_k: int = 50,
        top_p: float = 0.95,
):
    assert precision in ["fp32", "fp16", "bf16", "int8"]
    assert mode in ["cli", "gradio"]

    default_generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
        "top_p": top_p,
    }

    model = Huozi(model_name_or_path, precision)
    run_cli(default_generate_kwargs, model) if mode == "cli" else run_gradio(default_generate_kwargs, model)


if __name__ == "__main__":
    fire.Fire(main)
