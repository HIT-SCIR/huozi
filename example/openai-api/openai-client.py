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
