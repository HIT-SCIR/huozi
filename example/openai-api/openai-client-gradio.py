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

gr.ChatInterface(predict, chatbot=gr.Chatbot(height=500)).queue().launch()
