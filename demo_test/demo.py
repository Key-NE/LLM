from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import ModelScopeEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline


#推理模型初始化
model_path = r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").half().to(device)
pipe = pipeline(
    # "text-generation",
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True

)
llama_model = HuggingFacePipeline(pipeline=pipe)


#向量模型初始化
model_id = r"E:\llm\deepseek\emb_model"
embeddings = ModelScopeEmbeddings(model_id=model_id)


documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

vector_store = Chroma.from_documents(documents, embedding=embeddings)
retriever = RunnableLambda(vector_store.similarity_search).bind(k=2)


message = """
使用提供的上下文仅回答这个问题:
{question}
上下文:
{context}
"""




prompt_temp = ChatPromptTemplate.from_messages([('human', message)])
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | llama_model
print(chain.invoke('请介绍一下猫？请用中文回答'))
resp = chain.invoke('请介绍一下猫？请用中文回答')
# prompt_temp = ChatPromptTemplate.from_messages([('human', message)])
#
# # RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model。
# chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model
#
# resp = chain.invoke('请介绍一下猫？')
#
# print(resp.content)
