from bs4 import SoupStrainer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.document_loaders.parsers.html import bs4
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langserve import add_routes
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

tokenizer_chat = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")
model_chat = AutoModelForCausalLM.from_pretrained(r"E:\llm\deepseek\DeepSeek-R1-Distill-Qwen-1.5B")

#加载数据
# 定义过滤条件
# strainer = SoupStrainer(class_=('post-header', 'post-title', 'post-content'))

# 加载数据
loader = WebBaseLoader(web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'])

# 加载文档
docs = loader.load()

# 手动过滤 HTML 内容
# for doc in docs:
#     soup = bs4.BeautifulSoup(doc.page_content, 'html.parser', parse_only=strainer)
#     doc.page_content = soup.get_text()

# print(docs)
print(len(docs))

# text = 'cnm aasdasdasd asd a dqw d wdq ef 2tgethr ytb y r bt vfq r qcwce  asdadqwd asdaasdasdasdasdaf f qa drq wqfsf fqwdqsdqdefwefwef wefwef '
#大文本切割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split = splitter.split_documents(docs)

print(split)
# for re in split:
#     print(re,end='***\n')

# tokenizer_emb = AutoTokenizer.from_pretrained(r"E:\llm\deepseek\BAAIbge_small_zh")
# model_emb = AutoModel.from_pretrained(r"E:\llm\deepseek\BAAIbge_small_zh")
# embeddings = HuggingFaceEmbeddings(
#     model_name=r"E:\llm\deepseek\BAAIbge_small_zh",
#     model_kwargs={"device": "cpu"}  # 根据实际情况调整设备
# )
# 定义自定义嵌入函数
# def custom_embed(texts):
#     inputs = tokenizer_emb(texts, padding=True, truncation=True, return_tensors="pt")
#     outputs = model_emb(**inputs)
#     return outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()

model_path = r"/deepseek/BAAIbge_small_zh"
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # 可以使用一个默认的模型名称
    cache_folder=model_path  # 指定本地模型路径作为缓存文件夹
)
#存储
vectorstore = Chroma.from_documents(documents=split,embedding=embeddings)
# vectorstore = Chroma.from_documents(
#     documents=split,
#     embedding_function=model_emb,  # 使用自定义嵌入函数
#     persist_directory="chroma_db"
# )


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


resp = chain2.invoke({"input":"what is Task Decomposition?"})


print(resp['answer'])


#创建一个子链
#子链的提示模板
contextualize_q_system_prompt = """
定一段聊天历史以及用户最新提出的问题（该问题可能会引用聊天历史中的内容），
请构建一个无需借助聊天历史就能被理解的独立问题。请勿回答该问题，若有需要则对其重新组织表述，否则直接按原样返回。
"""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [

        ('system',contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human","{input}"),
]
)


#创建子链
history_chain = create_history_aware_retriever(model_chat,retriever,retriever_history_temp)


#保存问答的历史记录
store = {}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#创建父链chain
chain = create_retrieval_chain(history_chain,chain1)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

#第一轮对话
resp1 = result_chain.invoke(
    {'inpuy':'what is Task Decomposition?'},
    config={'configurable':{'session_id':'zs123456'}}
)
print(resp1['answer'])


#第二轮对话
resp1 = result_chain.invoke(
    {'inpuy':'what is Task Decomposition?'},
    config={'configurable':{'session_id':'zs123456'}}
)
print(resp1['answer'])