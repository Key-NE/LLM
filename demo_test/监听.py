from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# 配置
api_key = ""
api_url = "http://0.0.0.0:8020/v1"
model = "deepseek"

llm = OpenAI(model_name=model, openai_api_key=api_key, openai_api_base=api_url)
chat_model = ChatOpenAI(model_name=model, openai_api_key=api_key, openai_api_base=api_url)

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

# 使用 invoke 替代 predict_messages
print("LLM 运行结果：")
print(llm.invoke(messages))

print("ChatModel 运行结果：")
print(chat_model.invoke(messages))