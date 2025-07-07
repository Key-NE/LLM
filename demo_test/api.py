from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage


# 修改为你自己配置的OPENAI_API_KEY
api_key = "Key"

# 修改为你启动api-for-open-llm项目所在的服务地址和端口
api_url = "https://localhost:8020/v1"

modal= "DeepSeek-R1-Distill-Qwen-7B"

llm = OpenAI(model_name=modal,openai_api_key=api_key,openai_api_base=api_url)

chat_model = ChatOpenAI(model_name=modal,openai_api_key=api_key,openai_api_base=api_url)
text = "What would be a good company name for a company that makes colorful socks?"

messages = [HumanMessage(content=text)]


#LLMs: this is a language model which takes a string as input and returns a string
print("llm运行结果如下：")

print(llm.predict_messages(messages))

#ChatModels: this is a language model which takes a list of messages as input and returns a message
print("ChatModels运行结果如下：")
print(chat_model.predict_messages(messages))