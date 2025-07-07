#注意这里端口号改了
inference_server_url = "http://localhost:11434/v1"
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek",
    openai_api_key="none",
    openai_api_base=inference_server_url,
    max_tokens=500,
    temperature=0,
)
# print(model)
re = model.invoke("who are you?")
