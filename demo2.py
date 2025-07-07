from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

# 模型路径
model_path = r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda:0")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half()

# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True
)

# 创建 HuggingFacePipeline 实例
llama_model = HuggingFacePipeline(pipeline=pipe)

# 定义提示模板
template = '''
#背景信息# 
你是一名知识丰富的导航助手，了解中国每一个地方的名胜古迹及旅游景点. 
{history}
#问题# 
游客:我想去{input}旅游，给我推荐一下值得玩的地方?
'''
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# 创建记忆对象
memory = ConversationBufferMemory()

# 定义 get_session_history 函数
def get_session_history(inputs):
    return memory.load_memory_variables(inputs)["history"]

# 创建可运行对象
chain = RunnableWithMessageHistory(
    RunnablePassthrough.assign(
        history=lambda x: get_session_history(x)
    ) | prompt | llama_model,
    get_session_history=get_session_history,
    memory=memory
)

# 第一次提问
place = "天津"
result = chain.invoke({"input": place})
print(f"关于去 {place} 旅游的推荐: {result}")

# 第二次提问，这里可以基于第一次的对话历史进行推理
place = "北京"
result = chain.invoke({"input": place})
print(f"关于去 {place} 旅游的推荐: {result}")

# 查看对话历史
print("对话历史:", memory.load_memory_variables({}))