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


def ask(model, tokenizer, prompt, CUDA_Device, max_tokens=4096):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>')
    ]
    input_ids = tokenizer([prompt],
                          return_tensors='pt',
                          add_special_tokens=False).input_ids.to(CUDA_Device)
    generated_input = {
        'input_ids': input_ids,
        'max_new_tokens': max_tokens,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.9,
        'repetition_penalty': 1.1,
        'eos_token_id': terminators,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_token_id': tokenizer.pad_token_id
    }

    generated_ids = model.generate(**generated_input)
    ans = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return ans


from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
def call_model(state: MessagesState,model):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    # Summarize the messages if the chat history reaches a certain size
    if len(message_history) >= 4:
        last_human_message = state["messages"][-1]
        # Invoke the model to generate conversation summary
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        summary_message = model.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        # Delete messages that we no longer want to show up
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Call the model with summary & response
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = model.invoke([system_message] + state["messages"])

    return {"messages": message_updates}


def main():
    # 初始化
    doc_file_path = r'E:\llm\deepseek\document_hc'
    store_path = r'E:\llm\deepseek\Data_vecstore\Aquila.faiss'
    Embedding_Model = 'bge-m3:567m'
    LLM_Model_name = 'deepseek-r1:7b'
    CUDA_Device = 'cuda:0'
    model = load_llm_ollama(model_path_name=LLM_Model_name)
    vector_store = load_or_create_vector_store(store_path, doc_file_path, model_name=Embedding_Model)
    memory = ConversationBufferWindowMemory(k=5)  # k=1,意味着只能记住最后1轮对话内容
    RAG_TEMPLATE = """
    您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话并保持答案简洁
    <context>
    {context}
    </context>
    回答以下问题:
    {question}"""
    retriever = vector_store.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    system_prompt = SystemMessage(
        "你是一个有用的 AI 助手。用一句话简洁地回答用户的问题。")


    messages = [system_prompt]
    while True:
        # qiz = input('请输入您的问题: ')
        user_message = HumanMessage(input("\nUser: "))
        if user_message.content.lower() == 'bye' or user_message.content.lower() == 'exit':
            print('Bye~')
            break

        else:
            messages.append(user_message)
        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, user_message, 6, 0.2)

        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('无法从保存的向量存储中找到限定的上下文,在没有上下文的情况下与LLM交谈')
            for chunk in model.stream(messages):
                print(chunk.content, end='', flush=True)
            print()
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
                # {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"session_id": "foo"}}
            ))
            # for chunk in chain.stream(messages):
            #     print(chunk.content, end='', flush=True)
            # print()


if __name__ == '__main__':
    main()
