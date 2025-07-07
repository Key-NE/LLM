from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence, RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory



import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

model_path = r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half().to(device)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=8192,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True

)

#聊天机器人案例
llama_model = HuggingFacePipeline(pipeline=pipe)



# 定义提示模板
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "你是一个乐于助人的助手，用{language}尽你所能回答所有问题"),
#      MessagesPlaceholder(variable_name= 'my_msg' )
# ])
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "用{language}回答我的问题"),
     MessagesPlaceholder(variable_name= 'my_msg' )
])
chain = prompt_template | llama_model

# chain =  llama_model

# 保存聊天的历史记录
store = {}   # 所有用户的聊天记录都保存到store。 key： sessionId,value:历史聊天对象


#此函数预期接受一个session_id并返回一个消息历史记录对象
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # 每次聊天时候发送msg的key

)

config = {'configurable': {'session_id': ' zs123'}}  #给当前会话定义一个sessionId


# 第一轮
resp = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='你好 我是KEY')],
        'language': '中文'
    },
    config = config
)


# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='请问我的名字叫什么')],
        'language': '中文'
    },
    config = config
)

print(resp)
print(resp2)

print(type(resp))
# print(resp.content)
# 第三轮

# for resp in do_message.stream({'my_msg':[HumanMessage(content='请问:我的名字是什么？')],
#                                'language' : '中文'
#                                },
#                               config = config):
#每一次resp都是一个token
print(resp,end="-")
