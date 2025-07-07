import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader,Docx2txtLoader

tokenizer_chat = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B")
model_chat = AutoModelForCausalLM.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-7B")

def data_loader(directory_path=r'E:\llm\deepseek\document_hc'):
    all_docs = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# model_chat = HuggingFaceChat
# 加载数据
# loader = WebBaseLoader(web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'])
# loader = DirectoryLoader(r'E:\llm\deepseek\document_hc', glob="**/*.txt")
# loader = DirectoryLoader(r'E:\llm\deepseek\document_hc', glob="**/*.{txt,docx}",loaders={".txt": TextLoader,".docx": Docx2txtLoader})
# loader = TextLoader(r'E:\llm\deepseek\document_hc\demo.txt')
# 加载文档
# docs = loader.load()

#大文本切割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split = splitter.split_documents(data_loader())


print(split)
model_path = r"/deepseek/BAAIbge_small_zh"
# embeddings = HuggingFaceEmbeddings(
#     model_name=r"E:\llm\deepseek\BAAIbge_small_zh",
#     cache_folder=model_path
# )
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"local_files_only": True}
)
#存储
vectorstore = Chroma.from_documents(documents=split,embedding=embeddings)

# print(vectorstore)
#检索器
retriever = vectorstore.as_retriever()

#创建提示模板
system_prompt = """
你是一个负责问答任务的助手。使用以下检索到的上下文内容来回答问题。如果你不知道答案，就说明你不知道。最多用三句话，保持回答简洁。\n
{context}
"""


prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录 模板
    [
    ("system",system_prompt),
    # MessagesPlaceholder("chat_history"),
    ("human","{input}"),
]
)


#得到chain
chain1 =  create_stuff_documents_chain(model_chat,prompt)
chain2 =  create_retrieval_chain(retriever,chain1)

# print(chain2.invoke({'input':'你好'}))

print(chain2.invoke({"input": "你好"}))
# chain3 = create_stuff_documents_chain(model_chat,prompt)
# print(chain3)
# resp = chain2.invoke({"input":"what is Task Decomposition?"})
#
#
# print(resp['answer'])
#

# #创建一个子链
# #子链的提示模板
# contextualize_q_system_prompt = """
# 定一段聊天历史以及用户最新提出的问题（该问题可能会引用聊天历史中的内容），
# 请构建一个无需借助聊天历史就能被理解的独立问题。请勿回答该问题，若有需要则对其重新组织表述，否则直接按原样返回。
# """
#
# retriever_history_temp = ChatPromptTemplate.from_messages(
#     [
#
#         ('system',contextualize_q_system_prompt),
#         MessagesPlaceholder('chat_history'),
#         ("human","{input}"),
# ]
# )
#
#
# #创建子链
# history_chain = create_history_aware_retriever(model_chat,retriever,retriever_history_temp)
#
#
# #保存问答的历史记录
# store = {}
# def get_session_history(session_id:str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
#
# #创建父链chain
# chain = create_retrieval_chain(history_chain,chain1)
#
# result_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key='input',
#     history_messages_key='chat_history',
#     output_messages_key='answer',
# )
#
# #第一轮对话
# resp1 = result_chain.invoke(
#     {'inpuy':'what is Task Decomposition?'},
#     config={'configurable':{'session_id':'zs123456'}}
# )
# print(resp1['answer'])
#
#
# #第二轮对话
# resp1 = result_chain.invoke(
#     {'inpuy':'what is Task Decomposition?'},
#     config={'configurable':{'session_id':'zs123456'}}
# )
# print(resp1['answer'])