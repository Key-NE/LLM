from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"



class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]





# 加载pdf文件并返回文本段
def load_single_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    if not pdf_reader:
        return None
    ret = ''
    for i, page in enumerate(pdf_reader.pages):
        txt = page.extractText()
        if txt:
            ret += txt
    return ret


# 文档加载器
def data_loader(directory_path=r'E:\llm\deepseek\document_hc'):
    all_docs = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file.endswith('.pdf'):
                loader = load_single_pdf(file_path)
            else:
                continue
            if isinstance(loader, str):
                # 处理pdf加载返回的文本
                from langchain.docstore.document import Document
                doc = Document(page_content=loader, metadata={"source": file_path})
                all_docs.append(doc)
            else:
                docs = loader.load()
                all_docs.extend(docs)
    return all_docs


# 将文本拆分为 docs 文档
def split_text(txt, chunk_size=300, overlap=30, directory_path=r'E:\llm\deepseek\document_hc'):
    if not txt:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_documents(data_loader(directory_path))
    return docs


def create_embeddings(model_name="bge-m3:567m"):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings


def load_llm_ollama(model_path_name="deepseek-r1:7b"):
    chat_model = ChatOllama(
        model=model_path_name,
        temperature=0.6,
        num_predict=512,
        streaming=True  # 开启流式输出
    )
    return chat_model


# 使用 Embeddings 嵌入模型将文档保存到向量知识库存储
def create_vector_store(docs, embeddings, store_path):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store


# 从文件加载向量知识库
def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(
            store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    else:
        return None


# 加载或者创建向量知识库
def load_or_create_vector_store(store_path, doc_file_path, model_name):
    embeddings = create_embeddings(model_name)
    vector_store = load_vector_store(store_path, embeddings)
    if not vector_store:
        docs = split_text(doc_file_path)
        vector_store = create_vector_store(docs, embeddings, store_path)
    return vector_store


# 从 vector store 查询上下文
def query_vector_store(vector_store, query, k=4, relevance_threshold=0.2):
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    context = [doc[0].page_content for doc in related_docs]
    return context


# 加载 llm(deepseek) and tokenizer
def load_llm(model_path, CUDA_Device):
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=CUDA_Device,
                                                 torch_dtype=torch.float16,
                                                 quantization_config=quant_config)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # 初始化
    doc_file_path = r'E:\llm\deepseek\document_hc'
    store_path = r'E:\llm\deepseek\Data_vecstore\Aquila.faiss'
    Embedding_Model = 'bge-m3:567m'
    LLM_Model_name = 'deepseek-r1:7b'

    model = load_llm_ollama(model_path_name=LLM_Model_name)
    vector_store = load_or_create_vector_store(store_path, doc_file_path, model_name=Embedding_Model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an assistant who's good at {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


    RAG_TEMPLATE = """
    您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁
    <history>
    {history}
    </history>
    <context>
    {context}
    </context>
    回答以下问题:
    {question}"""
    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    system_prompt = SystemMessage(
        "你是一个有用的 AI 助手。用一句话简洁地回答用户的问题。")

    RAG_TEMPLATE = """
    您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁
    <history>
    {history}
    </history>
    回答以下问题:
    {question}"""
    rag_prompt_context = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    while True:
        qiz = input('请输入您的问题: ')

        if qiz == 'bye' or qiz == 'exit':
            print('Bye~')
            break
        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, qiz, 6, 0.2)

        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('无法从保存的向量存储中找到限定的上下文,在没有上下文的情况下与LLM交谈')
            context_str = '\n'.join(context)

            chain = (
                    RunnablePassthrough()
                    | rag_prompt_context
                    | model
                    | StrOutputParser()
            )
            chain_with_history = RunnableWithMessageHistory(
                chain,
                # 使用示例中定义的 get_by_session_id 函数
                # 以上。
                get_by_session_id,
                input_messages_key="question",
                history_messages_key="history",
            )
            print(chain_with_history.invoke(  # noqa: T201
                {"question": qiz},
                config={"configurable": {"session_id": "foo"}}
            ))
            # for chunk in model.stream(qiz):
            #     print(chunk.content, end='', flush=True)
            # print()
        else:
            # context = '\n'.join(context)
            # chain = (
            #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
            #         | rag_prompt
            #         | model
            # )


            context_str = '\n'.join(context)
            def get_context(_):
                return context_str
            chain = (
                    {"context": get_context, "question": RunnablePassthrough()}
                    | rag_prompt
                    | model
                    | StrOutputParser()
            )
            chain_with_history = RunnableWithMessageHistory(
                chain,
                # 使用示例中定义的 get_by_session_id 函数
                # 以上。
                get_by_session_id,
                input_messages_key="question",
                history_messages_key="history",
            )
            print(chain_with_history.invoke(  # noqa: T201
                {"question": qiz},
                config={"configurable": {"session_id": "foo"}}
            ))
            # for chunk in chain.stream(qiz):
            #     print(chunk.content, end='', flush=True)
            # print()


if __name__ == '__main__':
    main()
